#include <iostream>
#include <fstream>
#include "pin.H"
using namespace std;

ofstream OutFile;

static UINT64 icount = 32 * 32;

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool",
    "o", "itrace.out", "specify output file name");

static unsigned char opc[5] = {'\0'};
static int popfd_flag = 0;

VOID printip(void *ip, UINT32 size, const string *s, CONTEXT * ctxtx) {
	popfd_flag = 0;
	OutFile << static_cast<void*>(ip) << " ";
	unsigned char * cur = (unsigned char *)ip;
	while(size--){
		OutFile << setw(2) << setfill('0') << hex << unsigned int(*cur);
		cur++;
	}
	OutFile << " --> " << s->c_str() << endl;
    icount--;
	if(icount <=0){
	    PIN_ExitApplication(0);
	}
	for(int i = 0; i < 2; i++){
		opc[i] = *(cur++);
	}
	if(*(unsigned char *)ip == 0x9d){
		PIN_SetContextReg(ctxtx, REG_INST_PTR, (ADDRINT)(cur-4));
		PIN_ExecuteAt(ctxtx);
		popfd_flag = 1;
	}
}

VOID check_opc(void *ip, UINT32 size, CONTEXT * ctxtx){
	unsigned char * cur = ((unsigned char *)ip) + size;
	int flag = 0;
	if(popfd_flag)
		return;
	for(int i = 0; i < 2; i++){
		if(opc[i] != cur[i]){
			flag = 1;
			break;
		}
	}
	if(flag){
		PIN_SetContextReg(ctxtx, REG_INST_PTR, (ADDRINT)(((unsigned char *)ip) + size));
		PIN_ExecuteAt(ctxtx);
	}
}


VOID Instruction(INS ins, VOID *v)
{
	if(0x100000000000 < INS_Address(ins))
		return;
	INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR)printip,
					IARG_INST_PTR ,
					IARG_UINT32, INS_Size(ins),
					IARG_PTR, new string(INS_Disassemble(ins)),
					IARG_CONTEXT,
					IARG_END);
	if(INS_HasFallThrough(ins)){
		INS_InsertCall(ins, IPOINT_AFTER, (AFUNPTR)check_opc,
						IARG_INST_PTR,
						IARG_UINT32, INS_Size(ins),
						IARG_CONTEXT,
						IARG_END);
	}

}

VOID Fini(INT32 code, VOID *v)
{
	OutFile.close();
}

INT32 Usage()
{
	OutFile << endl << "error" << endl;
	OutFile.close();
    return -1;
}

int main(int argc, char * argv[])
{
    if (PIN_Init(argc, argv)) return Usage();
	
	OutFile.open(KnobOutputFile.Value().c_str());
	
    INS_AddInstrumentFunction(Instruction, 0);

    PIN_AddFiniFunction(Fini, 0);
    
    PIN_StartProgram();
    
    return 0;
}