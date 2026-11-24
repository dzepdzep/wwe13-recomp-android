// Old-SDK (PPC_*) -> new-SDK (REX_*) macro bridge for hand-written hooks.
#pragma once
#include "generated/default/wwe13_pch.h"
#ifndef PPC_FUNC
#define PPC_FUNC(x) REX_FUNC(x)
#endif
#define PPC_LOAD_U8 REX_LOAD_U8
#define PPC_LOAD_U16 REX_LOAD_U16
#define PPC_LOAD_U32 REX_LOAD_U32
#define PPC_LOAD_U64 REX_LOAD_U64
#define PPC_STORE_U8 REX_STORE_U8
#define PPC_STORE_U16 REX_STORE_U16
#define PPC_STORE_U32 REX_STORE_U32
#define PPC_STORE_U64 REX_STORE_U64
#define PPC_QUERY_TIMEBASE REX_QUERY_TIMEBASE
