
#ifndef DEEPPL_DLL_EXPORT_H
#define DEEPPL_DLL_EXPORT_H

#ifdef DEEPPL_DLL_STATIC_DEFINE
#  define DEEPPL_DLL_EXPORT
#  define DEEPPL_DLL_NO_EXPORT
#else
#  ifndef DEEPPL_DLL_EXPORT
#    ifdef deepPLDLL_EXPORTS
/* We are building this library */
#      define DEEPPL_DLL_EXPORT __declspec(dllexport)
#    else
/* We are using this library */
#      define DEEPPL_DLL_EXPORT __declspec(dllimport)
#    endif
#  endif

#  ifndef DEEPPL_DLL_NO_EXPORT
#    define DEEPPL_DLL_NO_EXPORT 
#  endif
#endif

#ifndef DEEPPL_DLL_DEPRECATED
#  define DEEPPL_DLL_DEPRECATED __declspec(deprecated)
#endif

#ifndef DEEPPL_DLL_DEPRECATED_EXPORT
#  define DEEPPL_DLL_DEPRECATED_EXPORT DEEPPL_DLL_EXPORT DEEPPL_DLL_DEPRECATED
#endif

#ifndef DEEPPL_DLL_DEPRECATED_NO_EXPORT
#  define DEEPPL_DLL_DEPRECATED_NO_EXPORT DEEPPL_DLL_NO_EXPORT DEEPPL_DLL_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef DEEPPL_DLL_NO_DEPRECATED
#    define DEEPPL_DLL_NO_DEPRECATED
#  endif
#endif

#endif /* DEEPPL_DLL_EXPORT_H */
