#ifndef RELATIONS_H
#define RELATIONS_H

#include "logup.cuh"
#include "utils.cuh"


typedef LookupElementsBasic<20> BlakeG;
typedef LookupElementsBasic<35> BlakeRound;
typedef LookupElementsBasic<17> BlakeRoundSigma;
typedef LookupElementsBasic<20> Cube252;
typedef LookupElementsBasic<2> MemoryAddressToId;
typedef LookupElementsBasic<29> MemoryIdToBig;
typedef LookupElementsBasic<3> Opcodes;
typedef LookupElementsBasic<73> PartialEcMul;
typedef LookupElementsBasic<57> PedersenPointsTable;
typedef LookupElementsBasic<42> Poseidon3PartialRoundsChain;
typedef LookupElementsBasic<32> PoseidonFullRoundChain;
typedef LookupElementsBasic<31> PoseidonRoundKeys;
typedef LookupElementsBasic<1> RangeCheck_6;
typedef LookupElementsBasic<1> RangeCheck_8;
typedef LookupElementsBasic<1> RangeCheck_11;
typedef LookupElementsBasic<1> RangeCheck_12;
typedef LookupElementsBasic<1> RangeCheck_18;
typedef LookupElementsBasic<1> RangeCheck_18_B;
typedef LookupElementsBasic<1> RangeCheck_19;
typedef LookupElementsBasic<1> RangeCheck_19_B;
typedef LookupElementsBasic<1> RangeCheck_19_C;
typedef LookupElementsBasic<1> RangeCheck_19_D;
typedef LookupElementsBasic<1> RangeCheck_19_E;
typedef LookupElementsBasic<1> RangeCheck_19_F;
typedef LookupElementsBasic<1> RangeCheck_19_G;
typedef LookupElementsBasic<1> RangeCheck_19_H;
typedef LookupElementsBasic<2> RangeCheck_3_6;
typedef LookupElementsBasic<2> RangeCheck_4_3;
typedef LookupElementsBasic<2> RangeCheck_4_4;
typedef LookupElementsBasic<2> RangeCheck_5_4;
typedef LookupElementsBasic<2> RangeCheck_9_9;
typedef LookupElementsBasic<2> RangeCheck_9_9_B;
typedef LookupElementsBasic<2> RangeCheck_9_9_C;
typedef LookupElementsBasic<2> RangeCheck_9_9_D;
typedef LookupElementsBasic<2> RangeCheck_9_9_E;
typedef LookupElementsBasic<2> RangeCheck_9_9_F;
typedef LookupElementsBasic<2> RangeCheck_9_9_G;
typedef LookupElementsBasic<2> RangeCheck_9_9_H;
typedef LookupElementsBasic<3> RangeCheck_7_2_5;
typedef LookupElementsBasic<4> RangeCheck_3_6_6_3;
typedef LookupElementsBasic<4> RangeCheck_4_4_4_4;
typedef LookupElementsBasic<5> RangeCheck_3_3_3_3_3;
typedef LookupElementsBasic<10> RangeCheckFelt252Width27;
typedef LookupElementsBasic<7> VerifyInstruction;
typedef LookupElementsBasic<3> VerifyBitwiseXor_4;
typedef LookupElementsBasic<3> VerifyBitwiseXor_7;
typedef LookupElementsBasic<3> VerifyBitwiseXor_8;
typedef LookupElementsBasic<3> VerifyBitwiseXor_8_B;
typedef LookupElementsBasic<3> VerifyBitwiseXor_9;
typedef LookupElementsBasic<3> VerifyBitwiseXor_12;
typedef LookupElementsBasic<8> TripleXor32;

#endif // RELATIONS_H