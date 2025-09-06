// ---------------------------------------
// Stacker Runtime Library
// Safe helpers for dicts, arrays, and lazy defaults
// ---------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h> // For PRId64

// -----------------------------
// Array Representation
// -----------------------------
typedef struct {
    int64_t length;
    int64_t* data;
} StackerArray;

// Create a new array
static StackerArray* stacker_array_new(int64_t length) {
    StackerArray* arr = (StackerArray*)malloc(sizeof(StackerArray));
    if (!arr) return NULL;
    arr->length = length;
    arr->data = (int64_t*)calloc(length, sizeof(int64_t));
    if (!arr->data) {
        free(arr);
        return NULL;
    }
    return arr;
}

// Free an array
static void stacker_array_free(StackerArray* arr) {
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

// Print an array (for debugging / say rest)
static void stacker_array_print(StackerArray* arr) {
    if (!arr || !arr->data) {
        printf("NULL\n");
        return;
    }
    printf("[");
    for (int64_t i = 0; i < arr->length; i++) {
        printf("%" PRId64, arr->data[i]);
        if (i < arr->length - 1) printf(", ");
    }
    printf("]");
}


// -----------------------------
// Dict Representation
// -----------------------------
typedef struct {
    char* key;
    char* value;
} StackerDictEntry;

typedef struct {
    int64_t size;
    StackerDictEntry* entries;
} StackerDict;

// Create new dict
static StackerDict* stacker_dict_new(int64_t size) {
    StackerDict* dict = (StackerDict*)malloc(sizeof(StackerDict));
    if (!dict) return NULL;
    dict->size = size;
    dict->entries = (StackerDictEntry*)calloc(size, sizeof(StackerDictEntry));
    if (!dict->entries) {
        free(dict);
        return NULL;
    }
    for (int64_t i = 0; i < size; i++) {
        dict->entries[i].key = NULL;
        dict->entries[i].value = NULL;
    }
    // calloc initializes key/value to NULL
    return dict;
}



// Lookup a key (returns NULL if not found)
static const char* stacker_dict_lookup(StackerDict* dict, const char* key) {
    if (!dict || !key) return NULL;
    for (int64_t i = 0; i < dict->size; i++) {
        if (dict->entries[i].key && strcmp(dict->entries[i].key, key) == 0) {
            return dict->entries[i].value;
        }
    }
    return NULL;
}

// Insert or update a key
static void stacker_dict_set(StackerDict* dict, const char* key, const char* value) {
    if (!dict || !key || !value) return;
    for (int64_t i = 0; i < dict->size; i++) {
        if (dict->entries[i].key && strcmp(dict->entries[i].key, key) == 0) {
            free(dict->entries[i].value);
            dict->entries[i].value = _strdup(value);
            return;
        }
    }
    for (int64_t i = 0; i < dict->size; i++) {
        if (dict->entries[i].key == NULL) {
            dict->entries[i].key = _strdup(key);
            dict->entries[i].value = _strdup(value);
            return;
        }
    }
}


// -----------------------------
// Lazy Default Representation
// -----------------------------
typedef struct {
    int evaluated;
    int64_t cached_value;
    int64_t(*thunk)(void);
} StackerLazy;

// Create a lazy thunk
static StackerLazy* stacker_lazy_new(int64_t(*fn)(void)) {
    StackerLazy* lazy = (StackerLazy*)malloc(sizeof(StackerLazy));
    if (!lazy) return NULL;
    lazy->evaluated = 0;
    lazy->cached_value = 0;
    lazy->thunk = fn;
    return lazy;
}

// Force evaluation
static int64_t stacker_lazy_force(StackerLazy* lazy) {
    if (!lazy) return 0;
    if (!lazy->evaluated) {
        lazy->cached_value = lazy->thunk();
        lazy->evaluated = 1;
    }
    return lazy->cached_value;
}

// Free lazy value
static void stacker_lazy_free(StackerLazy* lazy) {
    if (lazy) free(lazy);
}

int main(void) {
    return 0;
}
