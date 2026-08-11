/**
 *  @brief AES-256 in counter mode and in Galois/counter mode.
 *  @file python/stringzilla/cipher.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

/**
 *  @brief The class a refused Galois/counter mode tag raises, derived from `ValueError`.
 *
 *  A forged tag is a value-domain failure, so a caller who only guards against malformed input still
 *  catches a forgery, while a caller who must tell a forgery from a malformed argument names this class.
 */
PyObject *AuthenticationErrorType = NULL;

/** @brief An expanded counter-mode schedule, wiped when the object is collected. */
typedef struct {
    PyObject ob_base;
    sz_aes256_key_t key;
} Aes256CtrKey;

/** @brief An expanded schedule plus the Galois hash powers, wiped when the object is collected. */
typedef struct {
    PyObject ob_base;
    sz_aes256_gcm_key_t key;
} Aes256GcmKey;

/** @brief A chunked authenticated seal, holding its own copy of the key. */
typedef struct {
    PyObject ob_base;
    sz_aes256_gcm_encryptor_t encryptor;
} Aes256GcmEncryptor;

/** @brief A chunked authenticated open, holding its own copy of the key. */
typedef struct {
    PyObject ob_base;
    sz_aes256_gcm_decryptor_t decryptor;
} Aes256GcmDecryptor;

/**
 *  @brief  Overwrites @p length bytes at @p start with zeros, in a way a compiler may not elide.
 *
 *  The C core exposes no wipe entry point, and a plain `memset` over storage that is dead immediately
 *  afterwards is precisely the store an optimizer is licensed to drop. Writing through a `volatile`
 *  pointer makes every byte an observable side effect, so no schedule survives in memory a freed object
 *  once held. This translation unit sits outside the header-only tier, so it may spell the loop out rather
 *  than route through a kernel bound by that tier's no-libc-symbols rule.
 */
static void sz_py_wipe_bytes(void *start, sz_size_t length) {
    volatile sz_u8_t *cursor = (volatile sz_u8_t *)start;
    while (length--) *cursor++ = 0;
}

/**
 *  @brief  Exports a string-like object and confirms it spans exactly @p expected bytes.
 *          On failure sets a Python exception naming @p role and returns 0.
 */
static int sz_py_export_exact_bytes(PyObject *object, sz_size_t expected, char const *role, sz_cptr_t *start) {
    sz_size_t length;
    if (!sz_py_export_string_like(object, start, &length)) {
        wrap_current_exception("The argument must be string-like");
        return 0;
    }
    if (length != expected) {
        PyErr_Format(PyExc_ValueError, "%s must be exactly %zd bytes, received %zd", role, (Py_ssize_t)expected,
                     (Py_ssize_t)length);
        return 0;
    }
    return 1;
}

/**
 *  @brief  Reads the single `secret` argument every key constructor takes, positionally or by name.
 *          On failure sets a Python exception naming @p type_name and returns NULL.
 */
static PyObject *sz_py_export_secret_argument(PyObject *args, PyObject *kwargs, char const *type_name) {
    Py_ssize_t const positional_args_count = PyTuple_Size(args);
    if (positional_args_count > 1) {
        PyErr_Format(PyExc_TypeError, "%s() takes at most 1 positional argument", type_name);
        return NULL;
    }
    PyObject *secret_obj = positional_args_count == 1 ? PyTuple_GET_ITEM(args, 0) : NULL;
    if (kwargs) {
        Py_ssize_t position = 0;
        PyObject *name, *value;
        while (PyDict_Next(kwargs, &position, &name, &value)) {
            if (PyUnicode_CompareWithASCIIString(name, "secret") != 0) {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", name);
                return NULL;
            }
            if (secret_obj) {
                PyErr_SetString(PyExc_TypeError, "secret specified twice");
                return NULL;
            }
            secret_obj = value;
        }
    }
    if (!secret_obj) PyErr_Format(PyExc_TypeError, "%s() missing the required `secret` argument", type_name);
    return secret_obj;
}

static void Aes256CtrKey_dealloc(Aes256CtrKey *self) {
    sz_py_wipe_bytes(&self->key, sizeof(self->key));
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Aes256CtrKey_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    sz_unused_(args);
    sz_unused_(kwds);
    Aes256CtrKey *self = (Aes256CtrKey *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    sz_u8_t const placeholder_secret[SZ_AES256_KEY_LENGTH] = {0};
    sz_aes256_key_init(&self->key, placeholder_secret);
    return (PyObject *)self;
}

static int Aes256CtrKey_init(Aes256CtrKey *self, PyObject *args, PyObject *kwargs) {
    PyObject *secret_obj = sz_py_export_secret_argument(args, kwargs, "Aes256CtrKey");
    if (!secret_obj) return -1;
    sz_cptr_t secret;
    if (!sz_py_export_exact_bytes(secret_obj, SZ_AES256_KEY_LENGTH, "secret", &secret)) return -1;
    sz_aes256_key_init(&self->key, (sz_u8_t const *)secret);
    return 0;
}

static PyObject *Aes256CtrKey_xor(PyObject *self_obj, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *args_names_tuple) {
    Aes256CtrKey *self = (Aes256CtrKey *)self_obj;

    // Parse arguments
    PyObject *text_obj = NULL;
    PyObject *nonce_obj = NULL;
    PyObject *offset_obj = NULL;

    // Get count of keyword arguments
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const total_args = positional_args_count + args_names_count;

    // Validate total argument count
    if (total_args < 2 || total_args > 3) {
        PyErr_SetString(PyExc_TypeError, "xor() expects `text`, `nonce`, and an optional `offset`");
        return NULL;
    }

    // Handle positional arguments
    if (positional_args_count >= 1) text_obj = args[0];
    if (positional_args_count >= 2) nonce_obj = args[1];
    if (positional_args_count >= 3) offset_obj = args[2];

    // Handle keyword arguments
    for (Py_ssize_t index = 0; index < args_names_count; ++index) {
        PyObject *const name = PyTuple_GetItem(args_names_tuple, index);
        PyObject *const value = args[positional_args_count + index];

        if (PyUnicode_CompareWithASCIIString(name, "text") == 0) {
            if (text_obj) {
                PyErr_SetString(PyExc_TypeError, "text specified twice");
                return NULL;
            }
            text_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "nonce") == 0) {
            if (nonce_obj) {
                PyErr_SetString(PyExc_TypeError, "nonce specified twice");
                return NULL;
            }
            nonce_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "offset") == 0) {
            if (offset_obj) {
                PyErr_SetString(PyExc_TypeError, "offset specified twice");
                return NULL;
            }
            offset_obj = value;
        }
        else {
            PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    // Validate all required arguments are provided
    if (!text_obj || !nonce_obj) {
        PyErr_SetString(PyExc_TypeError, "xor() missing required arguments");
        return NULL;
    }

    sz_string_view_t text;
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }
    sz_cptr_t nonce;
    if (!sz_py_export_exact_bytes(nonce_obj, SZ_AES256_NONCE_LENGTH, "nonce", &nonce)) return NULL;

    sz_u64_t byte_offset = 0;
    if (offset_obj) {
        if (!PyLong_Check(offset_obj)) {
            PyErr_SetString(PyExc_TypeError, "offset must be an integer");
            return NULL;
        }
        byte_offset = (sz_u64_t)PyLong_AsUnsignedLongLong(offset_obj);
        if (PyErr_Occurred()) return NULL;
    }

    PyObject *output_obj = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)text.length);
    if (!output_obj) return NULL;
    sz_aes256_ctr_xor(&self->key, (sz_u8_t const *)nonce, byte_offset, text.start, text.length,
                      (sz_ptr_t)PyBytes_AS_STRING(output_obj));
    return output_obj;
}

static void Aes256GcmKey_dealloc(Aes256GcmKey *self) {
    sz_py_wipe_bytes(&self->key, sizeof(self->key));
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Aes256GcmKey_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    sz_unused_(args);
    sz_unused_(kwds);
    Aes256GcmKey *self = (Aes256GcmKey *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    sz_u8_t const placeholder_secret[SZ_AES256_KEY_LENGTH] = {0};
    sz_aes256_gcm_key_init(&self->key, placeholder_secret);
    return (PyObject *)self;
}

static int Aes256GcmKey_init(Aes256GcmKey *self, PyObject *args, PyObject *kwargs) {
    PyObject *secret_obj = sz_py_export_secret_argument(args, kwargs, "Aes256GcmKey");
    if (!secret_obj) return -1;
    sz_cptr_t secret;
    if (!sz_py_export_exact_bytes(secret_obj, SZ_AES256_KEY_LENGTH, "secret", &secret)) return -1;
    sz_aes256_gcm_key_init(&self->key, (sz_u8_t const *)secret);
    return 0;
}

static PyObject *Aes256GcmKey_encrypt(PyObject *self_obj, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple) {
    Aes256GcmKey *self = (Aes256GcmKey *)self_obj;

    // Parse arguments
    PyObject *text_obj = NULL;
    PyObject *nonce_obj = NULL;
    PyObject *associated_obj = NULL;

    // Get count of keyword arguments
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const total_args = positional_args_count + args_names_count;

    // Validate total argument count
    if (total_args < 2 || total_args > 3) {
        PyErr_SetString(PyExc_TypeError, "encrypt() expects `text`, `nonce`, and optional `associated` data");
        return NULL;
    }

    // Handle positional arguments
    if (positional_args_count >= 1) text_obj = args[0];
    if (positional_args_count >= 2) nonce_obj = args[1];
    if (positional_args_count >= 3) associated_obj = args[2];

    // Handle keyword arguments
    for (Py_ssize_t index = 0; index < args_names_count; ++index) {
        PyObject *const name = PyTuple_GetItem(args_names_tuple, index);
        PyObject *const value = args[positional_args_count + index];

        if (PyUnicode_CompareWithASCIIString(name, "text") == 0) {
            if (text_obj) {
                PyErr_SetString(PyExc_TypeError, "text specified twice");
                return NULL;
            }
            text_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "nonce") == 0) {
            if (nonce_obj) {
                PyErr_SetString(PyExc_TypeError, "nonce specified twice");
                return NULL;
            }
            nonce_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "associated") == 0) {
            if (associated_obj) {
                PyErr_SetString(PyExc_TypeError, "associated specified twice");
                return NULL;
            }
            associated_obj = value;
        }
        else {
            PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    // Validate all required arguments are provided
    if (!text_obj || !nonce_obj) {
        PyErr_SetString(PyExc_TypeError, "encrypt() missing required arguments");
        return NULL;
    }

    sz_string_view_t text;
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }
    sz_cptr_t nonce;
    if (!sz_py_export_exact_bytes(nonce_obj, SZ_AES256_NONCE_LENGTH, "nonce", &nonce)) return NULL;

    sz_string_view_t associated;
    associated.start = SZ_NULL, associated.length = 0;
    if (associated_obj && associated_obj != Py_None &&
        !sz_py_export_string_like(associated_obj, &associated.start, &associated.length)) {
        wrap_current_exception("The associated argument must be string-like");
        return NULL;
    }

    PyObject *output_obj = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)text.length);
    if (!output_obj) return NULL;
    sz_u8_t tag[SZ_AES256_TAG_LENGTH];
    sz_aes256_gcm_encrypt(&self->key, (sz_u8_t const *)nonce, associated.start, associated.length, text.start,
                          text.length, (sz_ptr_t)PyBytes_AS_STRING(output_obj), tag);

    PyObject *tag_obj = PyBytes_FromStringAndSize((char const *)tag, SZ_AES256_TAG_LENGTH);
    if (!tag_obj) {
        Py_DECREF(output_obj);
        return NULL;
    }
    PyObject *pair_obj = PyTuple_New(2);
    if (!pair_obj) {
        Py_DECREF(tag_obj);
        Py_DECREF(output_obj);
        return NULL;
    }
    PyTuple_SET_ITEM(pair_obj, 0, output_obj);
    PyTuple_SET_ITEM(pair_obj, 1, tag_obj);
    return pair_obj;
}

static PyObject *Aes256GcmKey_decrypt(PyObject *self_obj, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple) {
    Aes256GcmKey *self = (Aes256GcmKey *)self_obj;

    // Parse arguments
    PyObject *text_obj = NULL;
    PyObject *nonce_obj = NULL;
    PyObject *tag_obj = NULL;
    PyObject *associated_obj = NULL;

    // Get count of keyword arguments
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const total_args = positional_args_count + args_names_count;

    // Validate total argument count
    if (total_args < 3 || total_args > 4) {
        PyErr_SetString(PyExc_TypeError, "decrypt() expects `text`, `nonce`, `tag`, and optional `associated` data");
        return NULL;
    }

    // Handle positional arguments
    if (positional_args_count >= 1) text_obj = args[0];
    if (positional_args_count >= 2) nonce_obj = args[1];
    if (positional_args_count >= 3) tag_obj = args[2];
    if (positional_args_count >= 4) associated_obj = args[3];

    // Handle keyword arguments
    for (Py_ssize_t index = 0; index < args_names_count; ++index) {
        PyObject *const name = PyTuple_GetItem(args_names_tuple, index);
        PyObject *const value = args[positional_args_count + index];

        if (PyUnicode_CompareWithASCIIString(name, "text") == 0) {
            if (text_obj) {
                PyErr_SetString(PyExc_TypeError, "text specified twice");
                return NULL;
            }
            text_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "nonce") == 0) {
            if (nonce_obj) {
                PyErr_SetString(PyExc_TypeError, "nonce specified twice");
                return NULL;
            }
            nonce_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "tag") == 0) {
            if (tag_obj) {
                PyErr_SetString(PyExc_TypeError, "tag specified twice");
                return NULL;
            }
            tag_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "associated") == 0) {
            if (associated_obj) {
                PyErr_SetString(PyExc_TypeError, "associated specified twice");
                return NULL;
            }
            associated_obj = value;
        }
        else {
            PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", name);
            return NULL;
        }
    }

    // Validate all required arguments are provided
    if (!text_obj || !nonce_obj || !tag_obj) {
        PyErr_SetString(PyExc_TypeError, "decrypt() missing required arguments");
        return NULL;
    }

    sz_string_view_t text;
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }
    sz_cptr_t nonce, tag;
    if (!sz_py_export_exact_bytes(nonce_obj, SZ_AES256_NONCE_LENGTH, "nonce", &nonce)) return NULL;
    if (!sz_py_export_exact_bytes(tag_obj, SZ_AES256_TAG_LENGTH, "tag", &tag)) return NULL;

    sz_string_view_t associated;
    associated.start = SZ_NULL, associated.length = 0;
    if (associated_obj && associated_obj != Py_None &&
        !sz_py_export_string_like(associated_obj, &associated.start, &associated.length)) {
        wrap_current_exception("The associated argument must be string-like");
        return NULL;
    }

    PyObject *output_obj = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)text.length);
    if (!output_obj) return NULL;
    sz_status_t const status = sz_aes256_gcm_decrypt(&self->key, (sz_u8_t const *)nonce, associated.start,
                                                     associated.length, text.start, text.length,
                                                     (sz_ptr_t)PyBytes_AS_STRING(output_obj), (sz_u8_t const *)tag);
    if (status != sz_success_k) {
        Py_DECREF(output_obj);
        PyErr_SetString(AuthenticationErrorType, "The tag does not authenticate this ciphertext");
        return NULL;
    }
    return output_obj;
}

/**
 *  @brief  Reads the `key` and `nonce` arguments both chunked types take, positionally or by name.
 *          On failure sets a Python exception naming @p type_name and returns 0.
 */
static int sz_py_export_key_and_nonce_arguments(PyObject *args, PyObject *kwargs, char const *type_name,
                                                Aes256GcmKey const **key, sz_cptr_t *nonce) {
    Py_ssize_t const positional_args_count = PyTuple_Size(args);
    if (positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "%s() takes at most 2 positional arguments", type_name);
        return 0;
    }
    PyObject *key_obj = positional_args_count >= 1 ? PyTuple_GET_ITEM(args, 0) : NULL;
    PyObject *nonce_obj = positional_args_count >= 2 ? PyTuple_GET_ITEM(args, 1) : NULL;

    if (kwargs) {
        Py_ssize_t position = 0;
        PyObject *name, *value;
        while (PyDict_Next(kwargs, &position, &name, &value)) {
            if (PyUnicode_CompareWithASCIIString(name, "key") == 0) {
                if (key_obj) {
                    PyErr_SetString(PyExc_TypeError, "key specified twice");
                    return 0;
                }
                key_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(name, "nonce") == 0) {
                if (nonce_obj) {
                    PyErr_SetString(PyExc_TypeError, "nonce specified twice");
                    return 0;
                }
                nonce_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", name);
                return 0;
            }
        }
    }

    if (!key_obj || !nonce_obj) {
        PyErr_Format(PyExc_TypeError, "%s() missing the required `key` and `nonce` arguments", type_name);
        return 0;
    }
    if (!PyObject_TypeCheck(key_obj, &Aes256GcmKeyType)) {
        PyErr_SetString(PyExc_TypeError, "key must be an Aes256GcmKey object");
        return 0;
    }
    if (!sz_py_export_exact_bytes(nonce_obj, SZ_AES256_NONCE_LENGTH, "nonce", nonce)) return 0;
    *key = (Aes256GcmKey const *)key_obj;
    return 1;
}

static void Aes256GcmEncryptor_dealloc(Aes256GcmEncryptor *self) {
    sz_py_wipe_bytes(&self->encryptor, sizeof(self->encryptor));
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Aes256GcmEncryptor_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    sz_unused_(args);
    sz_unused_(kwds);
    Aes256GcmEncryptor *self = (Aes256GcmEncryptor *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    sz_u8_t const placeholder_secret[SZ_AES256_KEY_LENGTH] = {0};
    sz_u8_t const placeholder_nonce[SZ_AES256_NONCE_LENGTH] = {0};
    sz_aes256_gcm_key_t placeholder_key;
    sz_aes256_gcm_key_init(&placeholder_key, placeholder_secret);
    sz_aes256_gcm_encryptor_init(&self->encryptor, &placeholder_key, placeholder_nonce);
    sz_py_wipe_bytes(&placeholder_key, sizeof(placeholder_key));
    return (PyObject *)self;
}

static int Aes256GcmEncryptor_init(Aes256GcmEncryptor *self, PyObject *args, PyObject *kwargs) {
    Aes256GcmKey const *key;
    sz_cptr_t nonce;
    if (!sz_py_export_key_and_nonce_arguments(args, kwargs, "Aes256GcmEncryptor", &key, &nonce)) return -1;
    sz_aes256_gcm_encryptor_init(&self->encryptor, &key->key, (sz_u8_t const *)nonce);
    return 0;
}

static PyObject *Aes256GcmEncryptor_associate(PyObject *self_obj, PyObject *arg) {
    Aes256GcmEncryptor *self = (Aes256GcmEncryptor *)self_obj;
    sz_string_view_t text;
    if (!sz_py_export_string_like(arg, &text.start, &text.length)) {
        wrap_current_exception("Argument must be string-like");
        return NULL;
    }
    sz_aes256_gcm_encryptor_associate(&self->encryptor, text.start, text.length);
    Py_INCREF(self_obj);
    return self_obj;
}

static PyObject *Aes256GcmEncryptor_encrypt(PyObject *self_obj, PyObject *arg) {
    Aes256GcmEncryptor *self = (Aes256GcmEncryptor *)self_obj;
    sz_string_view_t text;
    if (!sz_py_export_string_like(arg, &text.start, &text.length)) {
        wrap_current_exception("Argument must be string-like");
        return NULL;
    }
    PyObject *output_obj = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)text.length);
    if (!output_obj) return NULL;
    sz_aes256_gcm_encryptor_update(&self->encryptor, text.start, text.length, (sz_ptr_t)PyBytes_AS_STRING(output_obj));
    return output_obj;
}

static PyObject *Aes256GcmEncryptor_digest(PyObject *self_obj, PyObject *noargs) {
    sz_unused_(noargs);
    Aes256GcmEncryptor *self = (Aes256GcmEncryptor *)self_obj;
    sz_u8_t tag[SZ_AES256_TAG_LENGTH];
    sz_aes256_gcm_encryptor_digest(&self->encryptor, tag);
    return PyBytes_FromStringAndSize((char const *)tag, SZ_AES256_TAG_LENGTH);
}

static void Aes256GcmDecryptor_dealloc(Aes256GcmDecryptor *self) {
    sz_py_wipe_bytes(&self->decryptor, sizeof(self->decryptor));
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Aes256GcmDecryptor_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    sz_unused_(args);
    sz_unused_(kwds);
    Aes256GcmDecryptor *self = (Aes256GcmDecryptor *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    sz_u8_t const placeholder_secret[SZ_AES256_KEY_LENGTH] = {0};
    sz_u8_t const placeholder_nonce[SZ_AES256_NONCE_LENGTH] = {0};
    sz_aes256_gcm_key_t placeholder_key;
    sz_aes256_gcm_key_init(&placeholder_key, placeholder_secret);
    sz_aes256_gcm_decryptor_init(&self->decryptor, &placeholder_key, placeholder_nonce);
    sz_py_wipe_bytes(&placeholder_key, sizeof(placeholder_key));
    return (PyObject *)self;
}

static int Aes256GcmDecryptor_init(Aes256GcmDecryptor *self, PyObject *args, PyObject *kwargs) {
    Aes256GcmKey const *key;
    sz_cptr_t nonce;
    if (!sz_py_export_key_and_nonce_arguments(args, kwargs, "Aes256GcmDecryptor", &key, &nonce)) return -1;
    sz_aes256_gcm_decryptor_init(&self->decryptor, &key->key, (sz_u8_t const *)nonce);
    return 0;
}

static PyObject *Aes256GcmDecryptor_associate(PyObject *self_obj, PyObject *arg) {
    Aes256GcmDecryptor *self = (Aes256GcmDecryptor *)self_obj;
    sz_string_view_t text;
    if (!sz_py_export_string_like(arg, &text.start, &text.length)) {
        wrap_current_exception("Argument must be string-like");
        return NULL;
    }
    sz_aes256_gcm_decryptor_associate(&self->decryptor, text.start, text.length);
    Py_INCREF(self_obj);
    return self_obj;
}

static PyObject *Aes256GcmDecryptor_decrypt_unverified(PyObject *self_obj, PyObject *arg) {
    Aes256GcmDecryptor *self = (Aes256GcmDecryptor *)self_obj;
    sz_string_view_t text;
    if (!sz_py_export_string_like(arg, &text.start, &text.length)) {
        wrap_current_exception("Argument must be string-like");
        return NULL;
    }
    PyObject *output_obj = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)text.length);
    if (!output_obj) return NULL;
    sz_aes256_gcm_decryptor_update_unverified(&self->decryptor, text.start, text.length,
                                              (sz_ptr_t)PyBytes_AS_STRING(output_obj));
    return output_obj;
}

static PyObject *Aes256GcmDecryptor_verify(PyObject *self_obj, PyObject *arg) {
    Aes256GcmDecryptor *self = (Aes256GcmDecryptor *)self_obj;
    sz_cptr_t tag;
    if (!sz_py_export_exact_bytes(arg, SZ_AES256_TAG_LENGTH, "tag", &tag)) return NULL;
    if (sz_aes256_gcm_decryptor_verify(&self->decryptor, (sz_u8_t const *)tag) != sz_success_k) {
        PyErr_SetString(AuthenticationErrorType, "The tag does not authenticate this ciphertext");
        return NULL;
    }
    Py_RETURN_NONE;
}

char const doc_AuthenticationError[] =                                                         //
    "Raised when a Galois/counter mode tag does not authenticate the bytes it accompanies.\n"  //
    "\n"                                                                                       //
    "Derives from `ValueError`, so code guarding only against malformed input still stops a\n" //
    "forgery, and the plaintext is never returned alongside it.\n"                             //
    "\n"                                                                                       //
    "Example:\n"                                                                               //
    "  >>> issubclass(sz.AuthenticationError, ValueError)\n"                                   //
    "  True";

static char const doc_Aes256CtrKey[] =                                                             //
    "Aes256CtrKey(secret)\n"                                                                       //
    "\n"                                                                                           //
    "AES-256 in counter mode, abbreviated CTR: unauthenticated, seekable, and its own inverse.\n"  //
    "The 32-byte secret is expanded once into a round-key schedule the object holds and wipes\n"   //
    "when it is collected.\n"                                                                      //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  secret (str | bytes): Exactly 32 secret bytes.\n"                                           //
    "Raises:\n"                                                                                    //
    "  ValueError: If the secret is not exactly 32 bytes long.\n"                                  //
    "Note:\n"                                                                                      //
    "  A nonce must never repeat under one secret; reuse exposes the exclusive-or of two texts.\n" //
    "\n"                                                                                           //
    "Example:\n"                                                                                   //
    "  >>> key = sz.Aes256CtrKey(bytes(32))\n"                                                     //
    "  >>> key.xor(key.xor(b'hello', bytes(12)), bytes(12))\n"                                     //
    "  b'hello'";

static char const doc_Aes256CtrKey_xor[] =                                                       //
    "xor(text, nonce, offset=0) -> bytes\n"                                                      //
    "\n"                                                                                         //
    "Exclusive-or `text` against the keystream, encrypting or decrypting with the same call.\n"  //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  text (str | bytes): The bytes to transform.\n"                                            //
    "  nonce (str | bytes): Exactly 12 nonce bytes, never repeated under one secret.\n"          //
    "  offset (int, optional): Absolute byte position of `text` in the stream. Defaults to 0.\n" //
    "Returns:\n"                                                                                 //
    "  bytes: As many bytes as `text` holds.\n"                                                  //
    "Raises:\n"                                                                                  //
    "  ValueError: If the nonce is not exactly 12 bytes long.\n"                                 //
    "\n"                                                                                         //
    "Example:\n"                                                                                 //
    "  >>> key, nonce = sz.Aes256CtrKey(bytes(32)), bytes(12)\n"                                 //
    "  >>> whole = key.xor(bytes(64), nonce)\n"                                                  //
    "  >>> key.xor(bytes(16), nonce, offset=32) == whole[32:48]\n"                               //
    "  True";

static char const doc_Aes256GcmKey[] =                                                              //
    "Aes256GcmKey(secret)\n"                                                                        //
    "\n"                                                                                            //
    "AES-256 in Galois/counter mode, abbreviated GCM: authenticated, and therefore not seekable.\n" //
    "The 32-byte secret is expanded once into a schedule and the Galois hash powers, also called\n" //
    "GHASH, and is wiped on collection.\n"                                                          //
    "\n"                                                                                            //
    "Args:\n"                                                                                       //
    "  secret (str | bytes): Exactly 32 secret bytes.\n"                                            //
    "Raises:\n"                                                                                     //
    "  ValueError: If the secret is not exactly 32 bytes long.\n"                                   //
    "Note:\n"                                                                                       //
    "  Reusing a nonce here exposes the hash subkey, after which every message can be forged.\n"    //
    "\n"                                                                                            //
    "Example:\n"                                                                                    //
    "  >>> key = sz.Aes256GcmKey(bytes(32))\n"                                                      //
    "  >>> ciphertext, tag = key.encrypt(b'hello', bytes(12))\n"                                    //
    "  >>> key.decrypt(ciphertext, bytes(12), tag)\n"                                               //
    "  b'hello'";

static char const doc_Aes256GcmKey_encrypt[] =                                                  //
    "encrypt(text, nonce, associated=None) -> (bytes, bytes)\n"                                 //
    "\n"                                                                                        //
    "Encrypt `text` and authenticate both it and `associated`, returning ciphertext and tag.\n" //
    "\n"                                                                                        //
    "Args:\n"                                                                                   //
    "  text (str | bytes): The plaintext.\n"                                                    //
    "  nonce (str | bytes): Exactly 12 nonce bytes, never repeated under one secret.\n"         //
    "  associated (str | bytes, optional): Bytes authenticated but not encrypted.\n"            //
    "Returns:\n"                                                                                //
    "  tuple: The ciphertext, as long as `text`, and the 16-byte tag.\n"                        //
    "Raises:\n"                                                                                 //
    "  ValueError: If the nonce is not exactly 12 bytes long.\n"                                //
    "\n"                                                                                        //
    "Example:\n"                                                                                //
    "  >>> ciphertext, tag = sz.Aes256GcmKey(bytes(32)).encrypt(b'hi', bytes(12), b'header')\n" //
    "  >>> len(ciphertext), len(tag)\n"                                                         //
    "  (2, 16)";

static char const doc_Aes256GcmKey_decrypt[] =                                               //
    "decrypt(text, nonce, tag, associated=None) -> bytes\n"                                  //
    "\n"                                                                                     //
    "Check `tag` against the ciphertext and associated data, then return the plaintext.\n"   //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  text (str | bytes): The ciphertext.\n"                                                //
    "  nonce (str | bytes): The 12 nonce bytes used to encrypt.\n"                           //
    "  tag (str | bytes): The 16-byte tag to check.\n"                                       //
    "  associated (str | bytes, optional): The same bytes passed to `encrypt`.\n"            //
    "Returns:\n"                                                                             //
    "  bytes: The plaintext, as long as `text`.\n"                                           //
    "Raises:\n"                                                                              //
    "  AuthenticationError: If the tag does not match, in which case nothing is returned.\n" //
    "\n"                                                                                     //
    "Example:\n"                                                                             //
    "  >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)\n"                             //
    "  >>> ciphertext, tag = key.encrypt(b'hello', nonce)\n"                                 //
    "  >>> key.decrypt(ciphertext, nonce, tag)\n"                                            //
    "  b'hello'";

static char const doc_Aes256GcmEncryptor[] =                                                       //
    "Aes256GcmEncryptor(key, nonce)\n"                                                             //
    "\n"                                                                                           //
    "The same Galois/counter mode construction as `Aes256GcmKey`, sealing a message in chunks.\n"  //
    "Chunk boundaries are invisible to the result. Opening a message uses `Aes256GcmDecryptor`,\n" //
    "a separate type because the tag absorbs ciphertext in both directions, which is the output\n" //
    "when sealing and the input when opening. Copies the key, so it cannot outlive it.\n"          //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  key (Aes256GcmKey): The expanded key, copied into the encryptor.\n"                         //
    "  nonce (str | bytes): Exactly 12 nonce bytes, never repeated under one secret.\n"            //
    "\n"                                                                                           //
    "Example:\n"                                                                                   //
    "  >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)\n"                                   //
    "  >>> sealed = sz.Aes256GcmEncryptor(key, nonce)\n"                                           //
    "  >>> sealed.encrypt(b'hel') + sealed.encrypt(b'lo') == key.encrypt(b'hello', nonce)[0]\n"    //
    "  True";

static char const doc_Aes256GcmEncryptor_associate[] =                                       //
    "associate(data) -> Aes256GcmEncryptor\n"                                                //
    "\n"                                                                                     //
    "Absorb bytes that are authenticated but not encrypted, and return self for chaining.\n" //
    "All associated data must be absorbed before the first message chunk.\n"                 //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  data (str | bytes): Associated bytes, such as a routing header.\n"                    //
    "Returns:\n"                                                                             //
    "  Aes256GcmEncryptor: The same object, enabling `sealed.associate(a).associate(b)`.\n"  //
    "Example:\n"                                                                             //
    "  >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)\n"                             //
    "  >>> sealed = sz.Aes256GcmEncryptor(key, nonce).associate(b'head')\n"                  //
    "  >>> sealed.encrypt(b'body') == key.encrypt(b'body', nonce, b'head')[0]\n"             //
    "  True";

static char const doc_Aes256GcmEncryptor_encrypt[] =                                //
    "encrypt(data) -> bytes\n"                                                      //
    "\n"                                                                            //
    "Encrypt one chunk and absorb its ciphertext into the running tag.\n"           //
    "\n"                                                                            //
    "Args:\n"                                                                       //
    "  data (str | bytes): The plaintext chunk.\n"                                  //
    "Returns:\n"                                                                    //
    "  bytes: The ciphertext chunk, as long as `data`.\n"                           //
    "Example:\n"                                                                    //
    "  >>> sealed = sz.Aes256GcmEncryptor(sz.Aes256GcmKey(bytes(32)), bytes(12))\n" //
    "  >>> len(sealed.encrypt(b'chunk'))\n"                                         //
    "  5";

static char const doc_Aes256GcmEncryptor_digest[] =                                          //
    "digest() -> bytes\n"                                                                    //
    "\n"                                                                                     //
    "Return the 16-byte tag over everything absorbed so far, leaving the seal appendable.\n" //
    "\n"                                                                                     //
    "Returns:\n"                                                                             //
    "  bytes: The 16-byte authentication tag.\n"                                             //
    "Example:\n"                                                                             //
    "  >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)\n"                             //
    "  >>> sealed = sz.Aes256GcmEncryptor(key, nonce)\n"                                     //
    "  >>> _ = sealed.encrypt(b'hello')\n"                                                   //
    "  >>> sealed.digest() == key.encrypt(b'hello', nonce)[1]\n"                             //
    "  True";

static char const doc_Aes256GcmDecryptor[] =                                                     //
    "Aes256GcmDecryptor(key, nonce)\n"                                                           //
    "\n"                                                                                         //
    "The same Galois/counter mode construction as `Aes256GcmKey`, opening a message in chunks\n" //
    "and checking its tag at the end. Chunk boundaries are invisible to the result. Sealing a\n" //
    "message uses `Aes256GcmEncryptor`. Copies the key, so it cannot outlive it.\n"              //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  key (Aes256GcmKey): The expanded key, copied into the decryptor.\n"                       //
    "  nonce (str | bytes): The 12 nonce bytes the message was sealed under.\n"                  //
    "\n"                                                                                         //
    "Example:\n"                                                                                 //
    "  >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)\n"                                 //
    "  >>> ciphertext, tag = key.encrypt(b'hello', nonce)\n"                                     //
    "  >>> sz.Aes256GcmDecryptor(key, nonce).decrypt_unverified(ciphertext)\n"                   //
    "  b'hello'";

static char const doc_Aes256GcmDecryptor_associate[] =                                       //
    "associate(data) -> Aes256GcmDecryptor\n"                                                //
    "\n"                                                                                     //
    "Absorb bytes that are authenticated but not encrypted, and return self for chaining.\n" //
    "All associated data must be absorbed before the first message chunk.\n"                 //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  data (str | bytes): The same associated bytes the sender authenticated.\n"            //
    "Returns:\n"                                                                             //
    "  Aes256GcmDecryptor: The same object, enabling `opened.associate(a).associate(b)`.\n"  //
    "Example:\n"                                                                             //
    "  >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)\n"                             //
    "  >>> ciphertext, tag = key.encrypt(b'body', nonce, b'head')\n"                         //
    "  >>> opened = sz.Aes256GcmDecryptor(key, nonce).associate(b'head')\n"                  //
    "  >>> opened.decrypt_unverified(ciphertext)\n"                                          //
    "  b'body'";

static char const doc_Aes256GcmDecryptor_decrypt_unverified[] =                                    //
    "decrypt_unverified(data) -> bytes\n"                                                          //
    "\n"                                                                                           //
    "Decrypt one chunk and absorb its ciphertext into the running tag. The bytes returned are\n"   //
    "not yet authenticated: nothing has checked that this ciphertext is genuine, so buffer them\n" //
    "until `verify()` succeeds rather than acting on them.\n"                                      //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  data (str | bytes): The ciphertext chunk.\n"                                                //
    "Returns:\n"                                                                                   //
    "  bytes: The unverified plaintext chunk, as long as `data`.\n"                                //
    "Example:\n"                                                                                   //
    "  >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)\n"                                   //
    "  >>> ciphertext, tag = key.encrypt(b'hello', nonce)\n"                                       //
    "  >>> opened = sz.Aes256GcmDecryptor(key, nonce)\n"                                           //
    "  >>> opened.decrypt_unverified(ciphertext)\n"                                                //
    "  b'hello'";

static char const doc_Aes256GcmDecryptor_verify[] =                                              //
    "verify(tag) -> None\n"                                                                      //
    "\n"                                                                                         //
    "Check `tag` against everything absorbed so far, in time independent of where it differs.\n" //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  tag (str | bytes): The 16-byte tag to check.\n"                                           //
    "Raises:\n"                                                                                  //
    "  AuthenticationError: If the tag does not match.\n"                                        //
    "  ValueError: If the tag is not exactly 16 bytes long.\n"                                   //
    "Example:\n"                                                                                 //
    "  >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)\n"                                 //
    "  >>> ciphertext, tag = key.encrypt(b'hello', nonce)\n"                                     //
    "  >>> opened = sz.Aes256GcmDecryptor(key, nonce)\n"                                         //
    "  >>> _ = opened.decrypt_unverified(ciphertext)\n"                                          //
    "  >>> opened.verify(tag) is None\n"                                                         //
    "  True";

static PyMethodDef Aes256CtrKey_methods[] = {
    {"xor", (PyCFunction)Aes256CtrKey_xor, SZ_METHOD_FLAGS, doc_Aes256CtrKey_xor}, //
    {NULL, NULL, 0, NULL},
};

PyTypeObject Aes256CtrKeyType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Aes256CtrKey",
    .tp_doc = doc_Aes256CtrKey,
    .tp_basicsize = sizeof(Aes256CtrKey),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Aes256CtrKey_new,
    .tp_init = (initproc)Aes256CtrKey_init,
    .tp_dealloc = (destructor)Aes256CtrKey_dealloc,
    .tp_methods = Aes256CtrKey_methods,
};

static PyMethodDef Aes256GcmKey_methods[] = {
    {"encrypt", (PyCFunction)Aes256GcmKey_encrypt, SZ_METHOD_FLAGS, doc_Aes256GcmKey_encrypt}, //
    {"decrypt", (PyCFunction)Aes256GcmKey_decrypt, SZ_METHOD_FLAGS, doc_Aes256GcmKey_decrypt}, //
    {NULL, NULL, 0, NULL},
};

PyTypeObject Aes256GcmKeyType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Aes256GcmKey",
    .tp_doc = doc_Aes256GcmKey,
    .tp_basicsize = sizeof(Aes256GcmKey),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Aes256GcmKey_new,
    .tp_init = (initproc)Aes256GcmKey_init,
    .tp_dealloc = (destructor)Aes256GcmKey_dealloc,
    .tp_methods = Aes256GcmKey_methods,
};

static PyMethodDef Aes256GcmEncryptor_methods[] = {
    {"associate", (PyCFunction)Aes256GcmEncryptor_associate, METH_O, doc_Aes256GcmEncryptor_associate}, //
    {"encrypt", (PyCFunction)Aes256GcmEncryptor_encrypt, METH_O, doc_Aes256GcmEncryptor_encrypt},       //
    {"digest", (PyCFunction)Aes256GcmEncryptor_digest, METH_NOARGS, doc_Aes256GcmEncryptor_digest},     //
    {NULL, NULL, 0, NULL},
};

PyTypeObject Aes256GcmEncryptorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Aes256GcmEncryptor",
    .tp_doc = doc_Aes256GcmEncryptor,
    .tp_basicsize = sizeof(Aes256GcmEncryptor),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Aes256GcmEncryptor_new,
    .tp_init = (initproc)Aes256GcmEncryptor_init,
    .tp_dealloc = (destructor)Aes256GcmEncryptor_dealloc,
    .tp_methods = Aes256GcmEncryptor_methods,
};

static PyMethodDef Aes256GcmDecryptor_methods[] = {
    {"associate", (PyCFunction)Aes256GcmDecryptor_associate, METH_O, doc_Aes256GcmDecryptor_associate}, //
    {"decrypt_unverified", (PyCFunction)Aes256GcmDecryptor_decrypt_unverified, METH_O,
     doc_Aes256GcmDecryptor_decrypt_unverified},                                               //
    {"verify", (PyCFunction)Aes256GcmDecryptor_verify, METH_O, doc_Aes256GcmDecryptor_verify}, //
    {NULL, NULL, 0, NULL},
};

PyTypeObject Aes256GcmDecryptorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Aes256GcmDecryptor",
    .tp_doc = doc_Aes256GcmDecryptor,
    .tp_basicsize = sizeof(Aes256GcmDecryptor),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Aes256GcmDecryptor_new,
    .tp_init = (initproc)Aes256GcmDecryptor_init,
    .tp_dealloc = (destructor)Aes256GcmDecryptor_dealloc,
    .tp_methods = Aes256GcmDecryptor_methods,
};
