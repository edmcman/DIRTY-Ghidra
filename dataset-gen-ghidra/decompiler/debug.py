import os
import pickle
from sys import argv

from ghidra.app.decompiler import DecompInterface
from ghidra.program.model.data import Undefined
#from ghidra.app.util.bin.format.dwarf4.next import DWARFImportOptions, DWARFProgram
# changed in ghidra 11.1
from ghidra.app.util.bin.format.dwarf import DWARFImportOptions, DWARFProgram
from ghidra.util.task import ConsoleTaskMonitor

from collect import Collector
from ghidra_function import Function
from ghidra_types import TypeLib


class CollectDebug(Collector):
    """Class for collecting debug information"""

    def __init__(self):
        print("Initializing CollectDebug")
        self.functions: Dict[int, Function] = dict()
        super().__init__()

    def write_functions(self) -> None:
        """Dumps the collected functions to the file specified by the environment
        variable `FUNCTIONS`.
        """
        with open(os.environ["FUNCTIONS"], "wb") as functions_fh:
            pickle.dump(self.functions, functions_fh)
            functions_fh.flush()

    def activate(self, ctx) -> int:
        """Collects types, user-defined variables, and their locations"""

        print("Collecting vars and types.")
        # `ea` is the start address of a single function
        dwarf_options = DWARFImportOptions()
        dwarf_options.setOutputDIEInfo(True)
        monitor = ConsoleTaskMonitor()
        dwarf_program = DWARFProgram(currentProgram(), dwarf_options, monitor)

        decomp = DecompInterface()
        decomp.toggleSyntaxTree(False)
        decomp.openProgram(dwarf_program.getGhidraProgram())

        # Ghidra separates Variables from their Data info, populate typelib first
        # for data in currentProgram().getListing().getDefinedData(True):
        #     self.type_lib.add_ghidra_type(data)

        for f in currentProgram().getListing().getFunctions(True):

            if f.isThunk():
                continue

            # Decompile
            decomp_results = decomp.decompileFunction(f, 30, None)

            if not decomp_results.decompileCompleted():
                continue

            if decomp_results.getErrorMessage() != "":
                continue

            # Function info
            #all_var_names = pickle.load(open(getScriptArgs()[0], 'rb'))
            high_func = decomp_results.getHighFunction()
            lsm = high_func.getLocalSymbolMap()
            symbols = [v for v in lsm.getSymbols()]
            func_return = high_func.getFunctionPrototype().getReturnType()

            name: str = f.getName()
            self.type_lib.add_ghidra_type(func_return)
            return_type = TypeLib.parse_ghidra_type(func_return)

            #for symbol in symbols:
            #    print(symbol.getDataType().getDescription())

            arguments = self.collect_variables(
                f.getStackFrame().getFrameSize(), [v for v in symbols if v.isParameter()] # and v.getName() in all_var_names],
            )
            local_vars = self.collect_variables(
                f.getStackFrame().getFrameSize(), [v for v in symbols if not v.isParameter()] # and v.getName() in all_var_names],
            )

            raw_code = decomp_results.getCCodeMarkup().toString()

            self.functions[f.getEntryPoint().toString()] = Function(
                name=name,
                return_type=return_type,
                arguments=arguments,
                local_vars=local_vars,
                raw_code=raw_code,
            )

        self.write_type_lib()
        self.write_functions()
        return 1


# ida.auto_wait()
# if not ida.init_hexrays_plugin():
#     ida.load_plugin("hexrays")
#     ida.load_plugin("hexx64")
#     if not ida.init_hexrays_plugin():
#         print("Unable to load Hex-rays")
#         ida.qexit(1)
#     else:
#         print(f"Hex-rays version {ida.get_hexrays_version()}")

debug = CollectDebug()
debug.activate(None)
# ida.qexit(0)
