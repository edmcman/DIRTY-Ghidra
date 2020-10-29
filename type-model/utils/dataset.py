import glob
import json
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union

import _jsonnet
import numpy as np
import torch
import webdataset as wds
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from utils.code_processing import tokenize_raw_code
from utils.function import CollectedFunction, Function
from utils.variable import Location, Variable, location_from_json_key
from utils.dire_types import Struct, TypeLibCodec, TypeLib


class Example:
    def __init__(
        self,
        name: str,
        code_tokens: str,
        source: Dict[str, Mapping[Location, Set[Variable]]],
        target: Dict[str, Mapping[Location, Set[Variable]]],
        binary_file: str = "",
        valid: bool = True,
        raw_code: str = "",
    ):
        self.name = name
        self.code_tokens = code_tokens
        self.source = source
        self.target = target
        self.binary_file = binary_file
        self._is_valid = valid
        self.raw_code = raw_code

    @classmethod
    def from_json(cls, d: Dict):
        source = defaultdict(dict)
        for loc in ["a", "l"]:
            if not loc in d["source"]:
                continue
            for key, args in d["source"][loc].items():
                source[loc][location_from_json_key(key)] = {
                    Variable.from_json(arg) for arg in args
                }
        target = defaultdict(dict)
        for loc in ["a", "l"]:
            if not loc in d["target"]:
                continue
            for key, args in d["target"][loc].items():
                target[loc][location_from_json_key(key)] = {
                    Variable.from_json(arg) for arg in args
                }
        return cls(d["name"], d["code_tokens"], source, target)

    def to_json(self):
        assert self._is_valid
        source = defaultdict(dict)
        for loc in ["a", "l"]:
            for key, args in self.source[loc].items():
                source[loc][key.json_key()] = [arg.to_json() for arg in args]
        target = defaultdict(dict)
        for loc in ["a", "l"]:
            for key, args in self.target[loc].items():
                target[loc][key.json_key()] = [arg.to_json() for arg in args]
        return {
            "name": self.name,
            "code_tokens": self.code_tokens,
            "source": source,
            "target": target,
        }

    @classmethod
    def from_cf(cls, cf: CollectedFunction, **kwargs):
        """Convert from a decoded CollectedFunction"""
        raw_code = cf.decompiler.raw_code
        code_tokens = tokenize_raw_code(raw_code)
        name = cf.decompiler.name

        code_tokens_set = set(code_tokens)
        source_locals = Example.filter(cf.decompiler.local_vars, code_tokens_set)
        source_args = Example.filter(cf.decompiler.arguments, code_tokens_set)
        target_locals = Example.filter(
            cf.debug.local_vars, None, set(source_locals.keys())
        )
        target_args = Example.filter(cf.debug.arguments, None, set(source_args.keys()))

        valid = (
            name == cf.debug.name
            and set(source_args.keys()) == set(target_args.keys())
            and set(source_locals.keys()) == set(target_locals.keys())
            and len(source_args) + len(source_locals) > 0
            and len(code_tokens) < 500
        )

        return cls(
            name,
            code_tokens,
            {"a": source_args, "l": source_locals},
            {"a": target_args, "l": target_locals},
            kwargs["binary_file"],
            valid,
            raw_code,
        )

    @staticmethod
    def filter(
        mapping: Mapping[Location, Set[Variable]],
        code_tokens: Optional[Set[str]] = None,
        locations: Optional[Set[Location]] = None,
    ):
        """Discard and leave these for future work:

        Register locations
        Locations are reused for multiple variables
        """
        ret: Mapping[Location, Set[Variable]] = {}
        for location, variable_set in mapping.items():
            if location.json_key().startswith("r"):
                continue
            if len(variable_set) > 1:
                continue
            if code_tokens and not list(variable_set)[0].name in code_tokens:
                continue
            if locations and not location in locations:
                continue
            ret[location] = variable_set
        return ret

    @property
    def is_valid_example(self):
        return self._is_valid

# HACK: Stupid global lambda functions to get distributed training working
def identity(x):
    return x
def get_key_json(x):
    return x['json']
def get_src_len(e):
    return e.source_seq_length

class Dataset(wds.Dataset):

    SHUFFLE_BUFFER = 5000
    SORT_BUFFER = 512

    def __init__(self, url: str, config: Optional[Dict] = None):
        # support wildcards
        urls = glob.glob(url)
        super().__init__(urls)
        if config:
            # annotate example for training
            from utils.vocab import Vocab

            self.vocab = Vocab.load(config["vocab_file"])
            with open(config["typelib_file"]) as type_f:
                self.typelib = TypeLibCodec.decode(type_f.read())
                self.typelib = self.typelib.fix()
            self.max_src_tokens_len = config["max_src_tokens_len"]
            assert not config["args"] # deal with local variables only for now
            self.locations = ["a", "l"] if config["args"] else ["l"]
            annotate = self._annotate
            sort = Dataset._sort
        else:
            # for creating the vocab
            annotate = identity
            sort = identity
            self.locations = ["a", "l"]
        self = (
            self.pipe(Dataset._file_iter_to_line_iter)
            .decode()
            .map(get_key_json)
            .map(Example.from_json)
            .map(annotate)
            .shuffle(Dataset.SHUFFLE_BUFFER)
            .pipe(sort)
        )

    @staticmethod
    def _sort(example_iter):
        sort_pool = []
        sort_pool_new = []
        for example in example_iter:
            if not example.valid:
                continue
            if sort_pool:
                yield sort_pool[len(sort_pool_new)]
            sort_pool_new.append(example)
            if len(sort_pool_new) == Dataset.SORT_BUFFER:
                sort_pool_new.sort(key=get_src_len)
                sort_pool = sort_pool_new
                sort_pool_new = []
        if sort_pool:
            yield from sort_pool[len(sort_pool_new):]
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
                yield {"json": line}

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
        src_var_names = []
        tgt_var_types = []
        tgt_var_type_objs = []
        for loc in self.locations:
            for key in sorted(example.source[loc], key=lambda x: x.offset):
                src_var = list(example.source[loc][key])[0]
                tgt_var = list(example.target[loc][key])[0]
                src_var_names.append(src_var.name)
                tgt_var_types.append(types_model[str(tgt_var.typ)])
                tgt_var_type_objs.append(tgt_var.typ)
                # HACK: Fix the bit/byte inconsistency bug in data
                if isinstance(src_var.typ, Struct):
                    TypeLib.fix_bit(src_var.typ)
                if isinstance(tgt_var.typ, Struct):
                    TypeLib.fix_bit(tgt_var.typ)

        valid = all([m.size <= 1024 for m in tgt_var_type_objs])
        if valid:
            src_a, src_s = Function.stack_layout(example.source["l"])
            tgt_a, tgt_s = Function.stack_layout(example.target["l"])
            if len(example.source["l"]) > 0 and src_a == tgt_a and src_s == tgt_s and tgt_a and tgt_a[-1] <= 1024:
                for typ in tgt_var_type_objs:
                    if not str(typ) in types_model:
                        valid = False
                        break
                valid &= self.typelib.valid_layout_for_types(tgt_a, tgt_s, tgt_var_type_objs)
            else:
                valid = False
        if valid:
            setattr(example, "valid", True)
            setattr(example, "src_var_names", src_var_names)
            setattr(example, "tgt_var_types", tgt_var_types)
            setattr(example, "mems", (tgt_a, tgt_s))
        else:
            setattr(example, "valid", False)

        return example

    @staticmethod
    def collate_fn(examples: List[Example]) -> Tuple[Dict[str, Union[torch.Tensor, int]], Dict[str, Union[torch.Tensor, List]]]:
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

        type_ids = [torch.tensor(e.tgt_var_types, dtype=torch.long) for e in examples]
        target_type_id = pad_sequence(type_ids, batch_first=True)
        assert target_type_id.shape == variable_mention_num.shape

        return (
            dict(
                src_code_tokens=input,
                variable_mention_to_variable_id=variable_mention_to_variable_id,
                variable_mention_mask=variable_mention_mask,
                variable_mention_num=variable_mention_num,
                variable_encoding_mask=variable_encoding_mask,
                batch_size=len(examples),
            ),
            dict(
                target_type_id=target_type_id,
                target_mask=target_type_id > 0,
                target_mems=[e.mems for e in examples]
            ),
        )
    
    def __len__(self):
        """HACK: fake length for testing in pl"""
        return 10 ** 4


if __name__ == "__main__":
    config = json.loads(_jsonnet.evaluate_file('config.xfmr.jsonnet'))
    dataset = Dataset("data1/dev-*.tar", config["data"])
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=0, batch_size=64, collate_fn=Dataset.collate_fn
    )
    for x in tqdm(dataloader):
        pass