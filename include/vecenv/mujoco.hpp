// The one place MuJoCo's header is included.
//
// mujoco.h pulls in mjsan.h, which exists for building MuJoCo ITSELF under
// AddressSanitizer: when the compiler defines __SANITIZE_ADDRESS__ it declares
// inline stack-marking helpers with an attribute syntax that g++ rejects in a
// C++ translation unit ("attributes are not allowed on a function-definition").
// This repo links the prebuilt, uninstrumented libmujoco from the wheel and
// never touches the stack helpers, so under our own sanitizer builds mjsan.h
// is skipped by pre-defining its include guard and the two stub macros it
// would otherwise provide. Without a sanitizer this header is a plain include.
#pragma once

#ifndef MUJOCO_INCLUDE_MJSAN_H_
#define MUJOCO_INCLUDE_MJSAN_H_
#define ASAN_POISON_MEMORY_REGION(addr, size) ((void)(addr), (void)(size))
#define ASAN_UNPOISON_MEMORY_REGION(addr, size) ((void)(addr), (void)(size))
#endif

#include <mujoco/mujoco.h>
