# Run inference on current function using DIRTY
from functools import lru_cache
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.program.model.pcode import HighFunctionDBUtil
from ghidra.program.model.symbol import SourceType
from ghidra.program.model.data import (
    PointerDataType,
    ArrayDataType,
    StructureDataType,
    UnionDataType,
    TypedefDataType,
)
from ghidra.app.services import DataTypeManagerService
from ghidra.program.model.address import Address
import json
import random
from collections import defaultdict
import sys
import os
import _jsonnet
import pathlib
import tqdm

DIRTY_PATH = pathlib.Path(os.path.realpath(__file__)).parent.parent.resolve()

TYPELIB_PATH = os.path.join(DIRTY_PATH, "dirty", "data1", "typelib_complete.json")

DIRTY_CONFIG = os.path.join(DIRTY_PATH, "dirty", "multitask.xfmr.jsonnet")

MODEL_CHECKPOINT = os.path.join(DIRTY_PATH, "dirty", "data1", "model.ckpt")

# Allow loading from the dirty directories.

sys.path.append(os.path.join(DIRTY_PATH, "dirty"))

# Load dirty modules

import utils.ghidra_types
from utils.ghidra_function import Function, CollectedFunction
from utils.ghidra_types import TypeLib, TypeInfo
from utils.ghidra_variable import Location, Register, Stack, Variable

from model.model import TypeReconstructionModel

import utils.infer

debug = False


def abort(s):
    raise Exception(s)

codec = utils.ghidra_types.TypeLibCodec()
typelib = codec.decode(open(TYPELIB_PATH, "r").read())

name_to_type = {}


def all_typenames():
    for _size, typelist in typelib.items():
        for _freq, typeentry in typelist:
            yield str(typeentry)


def find_types_by_name(name):
    for size, typelist in typelib.items():
        for _freq, typeentry in typelist:
            typename = str(typeentry)
            if typename == name:
                yield typeentry
    return


def find_type_by_name(name):
    try:
        return next(find_types_by_name(name))
    except StopIteration:
        print(f"Unable to find type {name} in typelib. Hopefully it is a built-in!")
        return utils.ghidra_types.TypeInfo(name=name, size=0)


def find_type_in_ghidra_typemanager(name, dtm):
    if dtm is None:
        dtm = currentProgram().getDataTypeManager()

    al = dtm.getAllDataTypes()
    output_list = [dt for dt in al if dt.getName() == name]

    if len(output_list) > 0:
        return output_list[0]
    else:
        return None


def find_type_in_any_ghidra_typemanager(name):
    tool = state().getTool()
    if tool is not None:
        dtms = (
            state().getTool().getService(DataTypeManagerService).getDataTypeManagers()
        )
    else:
        dtms = [currentProgram().getDataTypeManager()]
    for dtm in dtms:
        output = find_type_in_ghidra_typemanager(name, dtm)
        if output is not None:
            return output
    return None


@lru_cache(maxsize=20000)
def build_ghidra_type(typelib_type):

    # First check our cache.  This is important for self-referential types
    # (e.g., linked lists).
    if str(typelib_type) in name_to_type:
        return name_to_type[str(typelib_type)]

    out = find_type_in_any_ghidra_typemanager(str(typelib_type))
    if out is not None:
        return out

    if type(typelib_type) == utils.ghidra_types.TypeInfo:

        print(f"WARNING: {typelib_type.name} is a TypeInfo type: {typelib_type.debug}")
        print(f"WARNING: Unable to find type {typelib_type.name} in Ghidra.")

        match typelib_type.size:
            case 1:
                return find_type_in_any_ghidra_typemanager("byte")
            case 4:
                return find_type_in_any_ghidra_typemanager("uint32_t")
            case 8:
                return find_type_in_any_ghidra_typemanager("uint64_t")
            case _:
                abort(
                    f"Unknown type with unusual size: {typelib_type.size} {typelib_type.name}"
                )

    elif type(typelib_type) == utils.ghidra_types.Array:
        element_type = build_ghidra_type(find_type_by_name(typelib_type.element_type))
        return ArrayDataType(
            element_type, typelib_type.nelements, typelib_type.element_size
        )
    elif type(typelib_type) == utils.ghidra_types.Pointer:
        target_type = build_ghidra_type(
            find_type_by_name(typelib_type.target_type_name)
        )
        return PointerDataType(target_type)
        # Make type.
    elif (
        type(typelib_type) == utils.ghidra_types.Struct
        or type(typelib_type) == utils.ghidra_types.Union
    ):
        new_struct = (
            StructureDataType(typelib_type.name, typelib_type.size)
            if type(typelib_type) == utils.ghidra_types.Struct
            else UnionDataType(typelib_type.name)
        )
        # We need to immediately make this available in case we have a self-referential type.
        name_to_type[str(typelib_type)] = new_struct
        offset = 0
        for member in (
            typelib_type.layout
            if type(typelib_type) == utils.ghidra_types.Struct
            else typelib_type.members
        ):
            if type(member) == utils.ghidra_types.UDT.Padding:
                # Don't do anything?
                pass
                # new_struct.insertAtOffset(offset, VoidDataType(), member.size)
            elif type(member) == utils.ghidra_types.UDT.Field:
                member_type = build_ghidra_type(find_type_by_name(member.type_name))
                if type(typelib_type) == utils.ghidra_types.Struct:
                    new_struct.insertAtOffset(
                        offset, member_type, member.size, member.name, ""
                    )
                elif type(typelib_type) == utils.ghidra_types.Union:
                    new_struct.add(member_type, member.size, member.name, "")
                else:
                    abort("Unknown member type: " + str(type(typelib_type)))
            else:
                abort("Unknown member type: " + str(type(member)))

            offset = offset + member.size

            # field_type = build_ghidra_type(find_type_by_name(field.type))
            # new_struct.add(field_type, field.name, field.comment)
        return new_struct
    elif type(typelib_type) == utils.ghidra_types.TypeDef:
        other_type = build_ghidra_type(find_type_by_name(typelib_type.other_type_name))
        return TypedefDataType(typelib_type.name, other_type)
    else:
        abort(f"Unknown type: {type(typelib_type)} {typelib_type}")


def do_infer(cf, ghidra_function, redecompile=False):

    output = {}

    config = json.loads(_jsonnet.evaluate_file(DIRTY_CONFIG))

    # Set wd so the model can find data1/vocab.bpe10000

    os.chdir(os.path.join(DIRTY_PATH, "dirty"))

    model = TypeReconstructionModel.load_from_checkpoint(
        checkpoint_path=MODEL_CHECKPOINT, config=config
    )
    model.eval()

    model_output, model_output_multi, example_info, other_outputs = utils.infer.infer(config, model, cf)

    output['model_output'] = model_output
    output['model_output_multi'] = model_output_multi
    output['other_info'] = {'example_info': example_info, 'other_outputs': other_outputs}

    # Set up the decompiler
    decompiler = DecompInterface()
    decompiler.openProgram(ghidra_function.getProgram())

    # Decompile the current function
    print("Decompiling function " + ghidra_function.getName() + "...")
    results = decompiler.decompileFunction(ghidra_function, 0, ConsoleTaskMonitor())
    if not results.decompileCompleted():
        abort("Decompilation failed.")

    output["original_decompile"] = results.getDecompiledFunction().getC()

    # Get the high-level representation of the function
    high_function = results.getHighFunction()
    if not high_function:
        abort("Failed to get high-level function representation.")

    # Example: rename a specific variable (change the criteria as needed)
    for var in high_function.getLocalSymbolMap().getSymbols():

        original_name = var.getName()

        if original_name in model_output:
            new_type_name, new_name = model_output[original_name]
            if new_type_name != "disappear":

                if new_name in ["<unk>", "", "disappear"]:
                    new_name = original_name

                if new_name != original_name:
                    print("Renaming " + original_name + " to " + new_name + ".")

                new_type = None

                if new_type_name != "<unk>":
                    print(
                        f"Attempting to retype {original_name}/{new_name} to {new_type_name}"
                    )

                    try:
                        ti = find_type_by_name(new_type_name)
                        new_type = build_ghidra_type(ti)
                        print(
                            f"Changing type of {original_name}/{new_name} to {new_type_name}: {new_type}"
                        )
                    except Exception as e:
                        print(
                            f"Failed to find or build type {new_type_name} exception: {e}"
                        )

                try:
                    HighFunctionDBUtil.updateDBVariable(
                        var, new_name, new_type, SourceType.USER_DEFINED
                    )
                except Exception as e:
                    print(f"Failed to update variable {original_name} exception: {e}")

            else:
                print("Skipping disappear variable " + original_name + ".")
        else:
            print("No new name/type for " + original_name + " in prediction.")

    if redecompile:

        addrSet = ghidra_function.getBody()
        codeUnits = currentProgram().getListing().getCodeUnits(addrSet, True)
        asm = ""
        for codeUnit in codeUnits:
            asm += f"{hex(codeUnit.getAddress().getOffset())}: {codeUnit.toString()}\n"
        output["disassembly"] = asm


        results = decompiler.decompileFunction(ghidra_function, 0, ConsoleTaskMonitor())
        if not results.decompileCompleted():
            abort("Re-decompilation failed.")
        output["decompile"] = results.getDecompiledFunction().getC()

    return output


if sys.version_info.major < 3:
    abort(
        "You are not running Python 3.  This is probably a sign that you did not correctly configure Ghidrathon."
    )

if not isRunningHeadless():

    current_location = currentLocation()

    # Get the function containing this location.
    ghidra_function = getFunctionContaining(current_location.getAddress())

    assert ghidra_function is not None

    cf = utils.infer.ghidra_obtain_cf(ghidra_function)
    do_infer(cf, ghidra_function)

else:

    print("We are in headless mode.")

    args = getScriptArgs()
    outfile = args[0] if len(args) > 0 else "infer_success.txt"

    # Argument 0 is the output file for infer_success.txt.  This is used by the
    # CI.  Argument 1 is the target function to infer, if present.  This is used
    # by the huggingface space.
    targetFunAddr = hex(int(args[1])) if len(args) >= 2 else None

    function_manager = currentProgram().getFunctionManager()

    if targetFunAddr is not None:  # Huggingface space

        try:
            print(f"HF mode: {targetFunAddr}")
            addr = currentProgram().getAddressFactory().getAddress(targetFunAddr)
            print(f"Address: {addr}")
            fun = function_manager.getFunctionAt(addr)
            assert fun is not None, f"Unable to find function {targetFunAddr}"

            cf = utils.infer.ghidra_obtain_cf(fun)
            infer_out = do_infer(cf, fun, redecompile=True)

            json_output = {**infer_out}

            json.dump(json_output, open(outfile, "w"))
        except Exception as e:
            json_output = {"exception": str(e)}
            print(
                f"{targetFunAddr} failed because {e.__class__.__name__}: {str(e)}"
            )
            json.dump(json_output, open(outfile, "w"))

    else:  # CI mode
        print("CI mode")
        # Get all functions as an iterator
        function_iter = function_manager.getFunctions(True)

        # Keep trying functions until we find one that works!  This is needed
        # because small/trivial functions will fail.
        for ghidra_function in tqdm.tqdm(function_iter):
            if ghidra_function.isThunk() or ghidra_function.isExternal():
                continue
            print(f"Trying {ghidra_function}")
            try:
                cf = utils.infer.ghidra_obtain_cf(ghidra_function)
                do_infer(cf, ghidra_function)
                print("Success!")

                open(outfile, "w").write("success")
                # break
            except Exception as e:
                print(
                    f"{ghidra_function} failed because {e.__class__.__name__}: {str(e)}, trying next function"
                )
                continue
