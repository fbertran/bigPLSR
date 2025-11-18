#pragma once
// Optional CBLAS fast path.
// Define -DBIGPLSR_USE_CBLAS in PKG_CPPFLAGS to try enabling it.
// We set BIGPLSR_HAVE_CBLAS (0/1) depending on header availability.

#if defined(BIGPLSR_USE_CBLAS)
#if defined(__APPLE__)
#if __has_include(<vecLib/cblas.h>)
extern "C" {
#include <vecLib/cblas.h>
}
#define BIGPLSR_HAVE_CBLAS 1
#else
// Header not present in SDK; fall back.
#undef BIGPLSR_USE_CBLAS
#define BIGPLSR_HAVE_CBLAS 0
#endif
#else
#if __has_include(<cblas.h>)
extern "C" {
#include <cblas.h>
}
#define BIGPLSR_HAVE_CBLAS 1
#else
#undef BIGPLSR_USE_CBLAS
#define BIGPLSR_HAVE_CBLAS 0
#endif
#endif
#else
#define BIGPLSR_HAVE_CBLAS 0
#endif
