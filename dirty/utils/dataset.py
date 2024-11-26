import glob
import json
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union
from collections import defaultdict

import _jsonnet
import torch
import webdataset as wds
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence

from utils.code_processing import tokenize_raw_code
from utils.ghidra_function import CollectedFunction
from utils.ghidra_variable import Location, Variable, Unknown, location_from_json_key, Register, Stack
from utils.ghidra_types import TypeLibCodec, Disappear
from utils.vocab import Vocab

class Example:
    def __init__(
        self,
        name: str,
        code_tokens: str,
        source: Mapping[Location, Set[Variable]],
        target: Mapping[Location, Set[Variable]],
        binary_file: str = "",
        valid: bool = True,
        raw_code: str = "",
        test_meta: Dict[str, Dict[str, bool]] = None,
        binary: str = None,
        other_info = None,
    ):
        self.name = name
        self.code_tokens = code_tokens
        self.source = source
        self.target = target
        self.binary_file = binary_file
        self._is_valid = valid
        self.raw_code = raw_code
        self.test_meta = test_meta
        self.binary = binary
        self.other_info = other_info

    @classmethod
    def from_json(cls, d: Dict):
        source = {
            location_from_json_key(loc): [Variable.from_json(var) for var in varlist]
            for loc, varlist in d["source"].items()
        }
        target = {
            location_from_json_key(loc): [Variable.from_json(var) for var in varlist]
            for loc, varlist in d["target"].items()
        }

        # It seems like other code assumes the number of source and target
        # variables are the same.
        assert len(source) == len(target), "Source and target have different lengths"

        return cls(
            d["name"],
            d["code_tokens"],
            source,
            target,
            test_meta=d.get("test_meta", None),
            binary=d.get("binary", None),
        )

    def to_json(self):
        assert self._is_valid
        source = {loc.json_key(): [var.to_json() for var in varlist] for loc, varlist in self.source.items()}
        target = {loc.json_key(): [var.to_json() for var in varlist] for loc, varlist in self.target.items()}
        return {
            "name": self.name,
            "code_tokens": self.code_tokens,
            "source": source,
            "target": target,
        }

    @classmethod
    def from_cf(cls, cf: CollectedFunction, prediction=False, **kwargs):
        """Convert from a decoded CollectedFunction.
        """
        use_disappear = prediction
        filter_dups = not prediction
        name = cf.decompiler.name
        raw_code = cf.decompiler.raw_code
        code_tokens = tokenize_raw_code(raw_code)

        source = {**cf.decompiler.local_vars}

        # Actually merge these correctly!
        for k, v in cf.decompiler.arguments.items():
            # v is a set
            if k in source:
                source[k].update(v)
            else:
                source[k] = v
        if hasattr(cf.debug, "local_vars"):
            target = {**cf.debug.local_vars}
            for k, v in cf.debug.arguments.items():
                # v is a set
                if k in target:
                    target[k].update(v)
                else:
                    target[k] = v
        else:
            target = {}

        # Remove variables that overlap on memory or don't appear in the code tokens
        source_code_tokens_set = set(code_tokens[code_tokens.index("{"):])

        source, source_filtered_out = Example.filter(source, source_code_tokens_set, filter_out_duplicate_locations=filter_dups)
        target, target_filtered_out = Example.filter(target, None, set(source.keys()), filter_non_user_names=True, filter_out_duplicate_locations=filter_dups)

        # Optionally assign type "Disappear" to variables not existing in the
        # ground truth.  EJS thinks this may be harmful since the model learns
        # to overzealously predict disappear.

        # Note: Need to copy source.keys() so we don't change the list while
        # iterating.
        for loc in list(source.keys()):
            if use_disappear:
                if loc not in target.keys():
                    target[loc] = [Variable(Disappear(), "disappear", False)] * len(source[loc])
            else:
                if loc in source.keys() and loc not in target.keys():
                    del source[loc]

        varnames = set()
        # Add special tokens to variable names
        for varlist in source.values():
            for var in varlist:
                varname = var.name
                varnames.add(varname)
        for idx in range(len(code_tokens)):
            if code_tokens[idx] in varnames:
                code_tokens[idx] = f"@@{code_tokens[idx]}@@"

        other_info = {
            'source_filtered': source_filtered_out,
            'target_filtered': target_filtered_out,
        }

        return cls(
            name,
            code_tokens,
            source,
            target,
            kwargs["binary_file"],
            valid=source and "halt_baddata" not in source_code_tokens_set,
            raw_code=raw_code,
            other_info=other_info
        )

    @staticmethod
    def filter(
        mapping: Mapping[Location, Set[Variable]],
        code_tokens: Optional[Set[str]] = None,
        locations: Optional[Set[Location]] = None,
        filter_non_user_names: bool = False,
        filter_out_duplicate_locations: bool = True
    ) -> Mapping[Location, Set[Variable]]:
        """Discard and leave these for future work:

        Multiple variables sharing a memory location (no way to determine ground truth);
        Variables not appearing in code (no way to get representation);
        Target variables not appearing in source (useless ground truth);
        """
        ret: Mapping[Location, List[Variable]] = defaultdict(list)

        filtered = set()

        for location, variable_set in mapping.items():
            for v in variable_set:
                filtered.add((location, v))
            if len(variable_set) > 1 and filter_out_duplicate_locations:
                print(f"Warning: Ignoring location {location} with multiple variables {variable_set}")
                continue

            for var in variable_set:
                if code_tokens is not None and not var.name in code_tokens:
                    continue
                if locations is not None and not location in locations:
                    continue
                if filter_non_user_names and not var.user:
                    continue
                filtered.remove((location, var))
                ret[location].append(var)
        return ret, {x.name: loc.json_key() for loc, x in filtered}

    @property
    def is_valid_example(self):
        return self._is_valid


# HACK: Stupid global lambda functions required for distributed data loading
def identity(x):
    return x


def get_src_len(e):
    return e.source_seq_length

class Dataset(IterableDataset):

    SHUFFLE_BUFFER = 5000
    SORT_BUFFER = 512

    def __init__(self, url: str, config: Optional[Dict] = None, percent: float = 1.0):
        # support wildcards
        urls = sorted(glob.glob(url))
        urls = urls[: int(percent * len(urls))]

        if config:
            # annotate example for training
            self.vocab = Vocab.load(config["vocab_file"])
            # do we need this?
            #with open(config["typelib_file"]) as type_f:
            #    self.typelib = TypeLibCodec.decode(type_f.read())
            self.max_src_tokens_len = config["max_src_tokens_len"]
            self.max_num_var = config["max_num_var"]
            annotate = self._annotate
            self.rename = config.get("rename", False)
            # sort = Dataset._sort
            sort = identity
        else:
            # for creating the vocab
            annotate = identity
            sort = identity

        if not urls:

            # Dummy dataset for utils.infer

            self.len = None
            self.wds = None

        else:

            self.wds = (
                wds.WebDataset(urls, empty_check=False, shardshuffle=True)
                .compose(Dataset._file_iter_to_line_iter)
                .map(Example.from_json)
                .map(annotate)
                .shuffle(Dataset.SHUFFLE_BUFFER)
                .compose(sort)
            )

            # Estimate size of dataset
            # XXX: Limit number of files we read or use a timer?  Right
            # now we use all of them.
            try:
                line_dataset = (
                    wds.WebDataset(urls, shardshuffle=False)
                    .compose(Dataset._file_iter_to_line_iter)
                )
                #print(f"URLs: {urls} dataset: {line_dataset}")
                #print("Estimating size of dataset...")
                self.len = sum(1 for _line in line_dataset)
            except:
                # This might fail if we create a dummy dataset, such as in
                # utils.infer.
                self.len = None

    def __len__(self):
        return self.len

    # We need this for IterableDataset
    def __iter__(self):
        return iter(self.wds)

    # Should we forward to webdataset?
    #def __getattr__(self, name):
    #    return getattr(self.wds, name)

    @staticmethod
    def _sort(example_iter):
        sort_pool = []
        sort_pool_new = []
        for example in example_iter:
            if sort_pool:
                yield sort_pool[len(sort_pool_new)]
            sort_pool_new.append(example)
            if len(sort_pool_new) == Dataset.SORT_BUFFER:
                sort_pool_new.sort(key=get_src_len)
                sort_pool = sort_pool_new
                sort_pool_new = []
        if sort_pool:
            yield from sort_pool[len(sort_pool_new) :]
        if sort_pool_new:
            sort_pool_new.sort(key=get_src_len)
            yield from sort_pool

    @staticmethod
    def _file_iter_to_line_iter(jsonl_iter):
        for jsonl in jsonl_iter:
            lines = jsonl["jsonl"].split(b"\n")
            for line in lines:
                if not line:
                    continue
                json_line = json.loads(line)
                json_line["binary"] = jsonl["__key__"][: jsonl["__key__"].index("_")]
                yield json_line

    def _annotate(self, example: Example):
        src_bpe_model = self.vocab.source_tokens.subtoken_model
        snippet = example.code_tokens
        snippet = " ".join(snippet)
        sub_tokens = (
            ["<s>"]
            + src_bpe_model.encode_as_pieces(snippet)[: self.max_src_tokens_len]
            + ["</s>"]
        )
        sub_token_ids = (
            [src_bpe_model.bos_id()]
            + src_bpe_model.encode_as_ids(snippet)[: self.max_src_tokens_len]
            + [src_bpe_model.eos_id()]
        )
        setattr(example, "sub_tokens", sub_tokens)
        setattr(example, "sub_token_ids", sub_token_ids)
        setattr(example, "source_seq_length", len(sub_tokens))

        types_model = self.vocab.types
        subtypes_model = self.vocab.subtypes
        src_var_names = []
        tgt_var_names = []
        src_var_types_id = []
        src_var_types_str = []
        tgt_var_types_id = []
        tgt_var_types_str = []
        tgt_var_subtypes = []
        tgt_var_type_sizes = []
        tgt_var_type_objs = []
        src_var_locs_encoded = []
        tgt_names = []

        locs = sorted(example.source.keys(), key=lambda loc: repr(loc))

        stack_pos = [x.offset for x in example.source.keys() if isinstance(x, Stack)]
        stack_start_pos = max(stack_pos) if stack_pos else None

        def var_loc_in_func(loc):
            # TODO: fix the magic number (1030) for computing vocabulary idx
            # TODO: add vocabulary for unknown locations?
            if isinstance(loc, Register):
                return 1030 + self.vocab.regs[loc.name]
            elif isinstance(loc, Unknown):
                return 2 # unknown
            else:
                from utils.vocab import VocabEntry

                return (
                    3 + stack_start_pos - loc.offset
                    if stack_start_pos - loc.offset < VocabEntry.MAX_STACK_SIZE
                    else 2
                )

        def for_src_var(loc, src_var):
            nonlocal src_var_names, src_var_types_id, src_var_types_str, src_var_locs_encoded
            src_var_names.append(f"@@{src_var.name}@@")
            src_var_types_id.append(types_model.lookup_decomp(str(src_var.typ)))
            src_var_types_str.append(str(src_var.typ))
            # Memory
            # 0: absolute location of the variable in the function, e.g.,
            #   for registers: Reg 56
            #   for stack: relative position to the first variable
            # 1: size of the type
            # 2, 3, ...: start offset of fields in the type

            src_var_locs_encoded.append(
                [var_loc_in_func(loc)]
                + types_model.encode_memory(
                    (src_var.typ.size,) + src_var.typ.start_offsets()
                )
            )

        def for_tgt_var(loc, tgt_var):
            nonlocal tgt_var_names, tgt_var_types_id, tgt_var_types_str, tgt_var_subtypes, tgt_var_type_sizes, tgt_var_type_objs, tgt_names
            tgt_var_names.append(f"@@{tgt_var.name}@@")
            tgt_var_types_id.append(types_model[str(tgt_var.typ)])
            tgt_var_types_str.append(str(tgt_var.typ))
            if types_model[str(tgt_var.typ)] == types_model.unk_id:
                subtypes = [subtypes_model.unk_id, subtypes_model["<eot>"]]
            else:
                subtypes = [subtypes_model[subtyp] for subtyp in tgt_var.typ.tokenize()]
            tgt_var_type_sizes.append(len(subtypes))
            tgt_var_subtypes += subtypes
            tgt_var_type_objs.append(tgt_var.typ)
            tgt_names.append(tgt_var.name)

        for loc in locs[: self.max_num_var]:
            for src_var in example.source[loc]:
                for_src_var(loc, src_var)
            for tgt_var in example.target[loc]:
                for_tgt_var(loc, tgt_var)

        setattr(example, "src_var_names", src_var_names)
        setattr(example, "tgt_var_names", tgt_var_names)
        if self.rename:
            setattr(
                example,
                "tgt_var_name_ids",
                [self.vocab.names[n[2:-2]] for n in tgt_var_names],
            )
        setattr(example, "src_var_types", src_var_types_id)
        setattr(example, "src_var_types_str", src_var_types_str)
        setattr(example, "src_var_locs", src_var_locs_encoded)
        setattr(example, "tgt_var_types", tgt_var_types_id)
        setattr(example, "tgt_var_types_str", tgt_var_types_str)
        setattr(example, "tgt_var_subtypes", tgt_var_subtypes)
        setattr(example, "tgt_var_type_sizes", tgt_var_type_sizes)

        return example

    @staticmethod
    def collate_fn(
        examples: List[Example],
    ) -> Tuple[
        Dict[str, Union[torch.Tensor, int]], Dict[str, Union[torch.Tensor, List]]
    ]:
        token_ids = [torch.tensor(e.sub_token_ids) for e in examples]
        input = pad_sequence(token_ids, batch_first=True)
        max_time_step = input.shape[1]
        # corresponding var_id of each token in sub_tokens
        variable_mention_to_variable_id = torch.zeros(
            len(examples), max_time_step, dtype=torch.long
        )
        # if each token in sub_tokens is a variable
        variable_mention_mask = torch.zeros(len(examples), max_time_step)
        # the number of mentioned times for each var_id
        variable_mention_num = torch.zeros(
            len(examples), max(len(e.src_var_names) for e in examples)
        )

        for e_id, example in enumerate(examples):
            var_name_to_id = {name: i for i, name in enumerate(example.src_var_names)}
            for i, sub_token in enumerate(example.sub_tokens):
                if sub_token in example.src_var_names:
                    var_id = var_name_to_id[sub_token]
                    variable_mention_to_variable_id[e_id, i] = var_id
                    variable_mention_mask[e_id, i] = 1.0
                    variable_mention_num[e_id, var_name_to_id[sub_token]] += 1
        # if mentioned for each var_id
        variable_encoding_mask = (variable_mention_num > 0).float()

        src_type_ids = [
            torch.tensor(e.src_var_types, dtype=torch.long) for e in examples
        ]
        src_type_id = pad_sequence(src_type_ids, batch_first=True)
        tgt_type_ids = [torch.tensor(e.tgt_var_types, dtype=torch.long) for e in examples]
        target_type_id = pad_sequence(tgt_type_ids, batch_first=True)
        assert target_type_id.shape == variable_mention_num.shape, f"{target_type_id.shape} != {variable_mention_num.shape}"

        subtype_ids = [
            torch.tensor(e.tgt_var_subtypes, dtype=torch.long) for e in examples
        ]
        target_subtype_id = pad_sequence(subtype_ids, batch_first=True)
        type_sizes = [
            torch.tensor(e.tgt_var_type_sizes, dtype=torch.long) for e in examples
        ]
        target_type_sizes = pad_sequence(type_sizes, batch_first=True)

        src_type_mask = src_type_id > 0
        tgt_type_mask = target_type_id > 0

        src_var_locs = [
            torch.tensor(mems, dtype=torch.long)
            for e in examples
            for mems in e.src_var_locs
        ]
        src_var_locs = pad_sequence(src_var_locs, batch_first=True)
        src_var_locs_unflattened = torch.zeros(
            *src_type_mask.shape, src_var_locs.size(-1), dtype=torch.long
        )
        src_var_locs_unflattened[src_type_mask] = src_var_locs
        src_var_locs = src_var_locs_unflattened

        # renaming task
        if hasattr(examples[0], "tgt_var_name_ids"):
            name_ids = [
                torch.tensor(e.tgt_var_name_ids, dtype=torch.long) for e in examples
            ]
            target_name_id = pad_sequence(name_ids, batch_first=True)
        else:
            target_name_id = None

        return (
            dict(
                index=sum(
                    [
                        [(e.binary, e.name, name) for name in e.src_var_names]
                        for e in examples
                    ],
                    [],
                ),
                src_code_tokens=input,
                variable_mention_to_variable_id=variable_mention_to_variable_id,
                variable_mention_mask=variable_mention_mask,
                variable_mention_num=variable_mention_num,
                variable_encoding_mask=variable_encoding_mask,
                #target_type_src_mems=target_type_src_mems,
                src_type_id=src_type_id,
                src_type_mask=src_type_mask,
                src_var_locs=src_var_locs,
                #target_submask=target_subtype_id > 0,
                #target_type_sizes=target_type_sizes,
            ),
            dict(
                tgt_var_names=sum([e.tgt_var_names for e in examples], []),
                target_type_id=target_type_id,
                target_name_id=target_name_id,
                target_subtype_id=target_subtype_id,
                target_type_mask=tgt_type_mask,
                test_meta=[e.test_meta for e in examples],
            ),
        )


if __name__ == "__main__":
    config = json.loads(_jsonnet.evaluate_file("config.xfmr.jsonnet"))
    dataset = Dataset("data1/dev-*.tar", config["data"])
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=8, batch_size=64, collate_fn=Dataset.collate_fn
    )
    for x in dataloader:
        pass
