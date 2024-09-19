# Run inference on current function using DIRTY
from functools import lru_cache
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.program.model.pcode import HighFunctionDBUtil
from ghidra.program.model.symbol import SourceType
from ghidra.program.model.data import PointerDataType, ArrayDataType, StructureDataType, UnionDataType, TypedefDataType
from ghidra.app.services import DataTypeManagerService
import json
import random
from collections import defaultdict
import sys
import os
import _jsonnet
import pathlib
import tqdm

DIRTY_PATH = pathlib.Path(os.path.realpath(__file__)).parent.parent.resolve()

#DIRTY_PATH = "/home/ed/Projects/DIRTY/DIRTY-ghidra"

TYPELIB_PATH = os.path.join(DIRTY_PATH, 'dirty', 'data1', 'typelib_complete.json')

DIRTY_CONFIG = os.path.join(DIRTY_PATH, 'dirty', "multitask.xfmr.jsonnet")

MODEL_CHECKPOINT = os.path.join(DIRTY_PATH, 'dirty', 'data1', 'model.ckpt')

# Allow loading from the dirty directories.

sys.path.append(os.path.join(DIRTY_PATH, 'dirty'))

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

# Specialized version of dataset-gen-ghidra/decompiler/dump_trees.py

# Specialized version from collect.py

def collect_variables(variables):

    collected_vars = defaultdict(set)

    for v in variables:
        if v.getName() == "":
            continue

        typ: TypeInfo = TypeLib.parse_ghidra_type(v.getDataType())

        loc: Optional[Location] = None
        storage = v.getStorage()

        if storage.isStackStorage():
            loc = Stack(storage.getStackOffset())
        elif storage.isRegisterStorage():
            loc = Register(storage.getRegister().getName())
        else:
            print(f"Unknown storage type for {v} {v.getName()}: {storage}")
        if loc is not None:
            collected_vars[loc].add(
                Variable(typ=typ, name=v.getName(), user=False)
            )
    return collected_vars

def dump(f):

    decomp = DecompInterface()
    decomp.toggleSyntaxTree(False)
    decomp.openProgram(currentProgram())

    decomp_results = decomp.decompileFunction(f, 30, None)
    #f = decomp_results.getFunction()

    if not decomp_results.decompileCompleted():
        abort("Failed to decompile")

    if decomp_results.getErrorMessage() != "":
        abort("Failed to decompile")

    high_func = decomp_results.getHighFunction()
    lsm = high_func.getLocalSymbolMap()
    symbols = [v for v in lsm.getSymbols()]
    func_return = high_func.getFunctionPrototype().getReturnType()

    name:str = f.getName()

    return_type = utils.ghidra_types.TypeLib.parse_ghidra_type(func_return)

    arguments = collect_variables(
        [v for v in symbols if v.isParameter()],
    )
    local_vars = collect_variables(
        [v for v in symbols if not v.isParameter()],
    )

    raw_code = decomp_results.getCCodeMarkup().toString()

    decompiler = Function(
        ast=None,
        name=name,
        return_type=return_type,
        arguments=arguments,
        local_vars=local_vars,
        raw_code=raw_code,
    )

    cf = CollectedFunction(
            ea=f.getEntryPoint().toString(),
            debug=None,
            decompiler=decompiler,
        )
    
    return cf

# End dump_trees.py

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
            #if size == 160000: # or "double" in typename:
            #if debug and name in typename:
            #    print(f"Hmm: {name} ==? {typename}")
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
        dtms = state().getTool().getService(DataTypeManagerService).getDataTypeManagers()
    else:
        dtms = [currentProgram().getDataTypeManager()]
    for dtm in dtms:
        output = find_type_in_ghidra_typemanager(name, dtm)
        if output is not None:
            return output
    return None

@lru_cache(maxsize=20000)
def build_ghidra_type(typelib_type):
    #print(f"build_ghidra_type {typelib_type} {typelib_type.__dict__} {type(typelib_type)}")

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
                abort(f"Unknown type with unusual size: {typelib_type.size} {typelib_type.name}")

    elif type(typelib_type) == utils.ghidra_types.Array:
        element_type = build_ghidra_type(find_type_by_name(typelib_type.element_type))
        return ArrayDataType(element_type, typelib_type.nelements, typelib_type.element_size)
    elif type(typelib_type) == utils.ghidra_types.Pointer:
        target_type = build_ghidra_type(find_type_by_name(typelib_type.target_type_name))
        return PointerDataType(target_type)
        # Make type.
    elif type(typelib_type) == utils.ghidra_types.Struct or type(typelib_type) == utils.ghidra_types.Union:
        new_struct = StructureDataType(typelib_type.name, typelib_type.size) if type(typelib_type) == utils.ghidra_types.Struct else UnionDataType(typelib_type.name)
        # We need to immediately make this available in case we have a self-referential type.
        name_to_type[str(typelib_type)] = new_struct
        offset = 0
        for member in (typelib_type.layout if type(typelib_type) == utils.ghidra_types.Struct else typelib_type.members):
            if type(member) == utils.ghidra_types.UDT.Padding:
                # Don't do anything?
                pass
                # new_struct.insertAtOffset(offset, VoidDataType(), member.size)
            elif type(member) == utils.ghidra_types.UDT.Field:
                member_type = build_ghidra_type(find_type_by_name(member.type_name))
                if type(typelib_type) == utils.ghidra_types.Struct:
                    new_struct.insertAtOffset(offset, member_type, member.size, member.name, "")
                elif type(typelib_type) == utils.ghidra_types.Union:
                    new_struct.add(member_type, member.size, member.name, "")
                else:
                    abort("Unknown member type: " + str(type(typelib_type)))
            else:
                abort("Unknown member type: " + str(type(member)))

            offset = offset + member.size
                
            #field_type = build_ghidra_type(find_type_by_name(field.type))
            #new_struct.add(field_type, field.name, field.comment)
        return new_struct
    elif type(typelib_type) == utils.ghidra_types.TypeDef:
        other_type = build_ghidra_type(find_type_by_name(typelib_type.other_type_name))
        return TypedefDataType(typelib_type.name, other_type)
    else:
        abort(f"Unknown type: {type(typelib_type)} {typelib_type}")

def test_types():
    total = 0
    succ = 0

    l = list(all_typenames())
    random.shuffle(l)

    #l = ["longlong[20][10]"]
    #l = ["xen_string_string_map"]
    #debug = True

    for typename in l:
        print(f"Trying to build type {typename}")
        total += 1

        if monitor().isCancelled():
            break

        monitor().setMessage(f"Building type {typename}")

        try:
            ti = find_type_by_name(typename)
        except Exception as e:
            print(f"Failed to find type {typename} exception: {e}")
            continue

        try:
            gtype = build_ghidra_type(ti)
            assert gtype is not None, "build_ghidra_type returned None."
            print(f"Successfully built type {typename} in Ghidra: {gtype}")
        except Exception as e:
            print(f"Failed to build ghidra type {typename} exception: {e}")
            #break
            continue

        succ += 1

        print(f"Successfully built {succ}/{total} {float(succ)/total} types.")

    exit(0)

#exeName = currentProgram().getName()

#jsonFile = askFile("Select JSON file", "Open")
#print("Parsing JSON")
#jsonObj = json.load(open(jsonFile.getAbsolutePath()))

#if exeName in jsonObj:
#    jsonObj = jsonObj[exeName]
#elif len(jsonObj) == 1:
#    jsonObj = jsonObj[list(jsonObj.keys())[0]]
#else:
#    abort("Unable to find the executable in the JSON file.")

def do_infer(cf):

    config = json.loads(_jsonnet.evaluate_file(DIRTY_CONFIG))

    # Set wd so the model can find data1/vocab.bpe10000

    os.chdir(os.path.join(DIRTY_PATH, 'dirty'))

    model = TypeReconstructionModel.load_from_checkpoint(checkpoint_path=MODEL_CHECKPOINT, config=config) 
    model.eval()

    model_output = utils.infer.infer(config, model, cf)
    print(model_output)

    # Set up the decompiler
    decompiler = DecompInterface()
    decompiler.openProgram(current_function.getProgram())

    # Decompile the current function
    print("Decompiling function " + current_function.getName() + "...")
    results = decompiler.decompileFunction(current_function, 0, ConsoleTaskMonitor())
    if not results.decompileCompleted():
        abort("Decompilation failed.")

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

                if new_name in ["<unk>", ""]:
                    new_name = original_name

                if new_name != original_name:
                    print("Renaming " + original_name + " to " + new_name + ".")

                new_type = None



                if new_type_name != "<unk>":
                    print(f"Attempting to retype {original_name}/{new_name} to {new_type_name}")

                    try:
                        ti = find_type_by_name(new_type_name)
                        new_type = build_ghidra_type(ti)
                        print(f"Changing type of {original_name}/{new_name} to {new_type_name}: {new_type}")
                    except Exception as e:
                        print(f"Failed to find or build type {new_type_name} exception: {e}")

                try:
                    HighFunctionDBUtil.updateDBVariable(var, new_name, new_type, SourceType.USER_DEFINED)
                except Exception as e:
                    print(f"Failed to update variable {original_name} exception: {e}")


            else:
                print("Skipping disappear variable " + original_name + ".")
        else:
            print("No new name/type for " + original_name + " in prediction.")

if not isRunningHeadless():

    current_location = currentLocation()

    # Get the function containing this location.
    current_function = getFunctionContaining(current_location.getAddress())

    assert current_function is not None

    cf = dump(current_function)
    do_infer(cf)

else:

    print("We are in headless mode.")

    function_manager = currentProgram().getFunctionManager()
    
    # Get all functions as an iterator
    function_iter = function_manager.getFunctions(True)

    # Keep trying functions until we find one that works!  This is needed
    # because small/trivial functions will fail.
    for current_function in tqdm.tqdm(function_iter):
        if current_function.isThunk() or current_function.isExternal():
            continue
        print(f"Trying {current_function}")
        try:
            cf = dump(current_function)
            do_infer(cf)
            print("Success!")
            args = getScriptArgs()
            outfile = args[0] if len(args) > 0 else "infer_success.txt"
            open(outfile, "w").write("success")
            #break
        except Exception as e:
            print(f"{current_function} because {e.__class__.__name__}: {str(e)}, trying next function")
            continue
