// ---------------------------------------
// Stacker Runtime Library
// Safe helpers for dicts, arrays, and lazy defaults
// ---------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// -----------------------------
// Array Representation
// -----------------------------
typedef struct {
    int64_t length;
    int64_t *data;
} StackerArray;

// Create a new array
StackerArray* stacker_array_new(int64_t length) {
    StackerArray* arr = (StackerArray*) malloc(sizeof(StackerArray));
    arr->length = length;
    arr->data = (int64_t*) calloc(length, sizeof(int64_t));
    return arr;
}

// Free an array
void stacker_array_free(StackerArray* arr) {
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

// Print an array (for debugging / say rest)
void stacker_array_print(StackerArray* arr) {
    printf("[");
    for (int64_t i = 0; i < arr->length; i++) {
        printf("%ld", arr->data[i]);
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
StackerDict* stacker_dict_new(int64_t size) {
    StackerDict* dict = (StackerDict*) malloc(sizeof(StackerDict));
    dict->size = size;
    dict->entries = (StackerDictEntry*) calloc(size, sizeof(StackerDictEntry));
    return dict;
}

// Free dict
void stacker_dict_free(StackerDict* dict) {
    if (dict) {
        for (int64_t i = 0; i < dict->size; i++) {
            free(dict->entries[i].key);
            free(dict->entries[i].value);
        }
        free(dict->entries);
        free(dict);
    }
}

// Lookup a key (returns NULL if not found)
const char* stacker_dict_lookup(StackerDict* dict, const char* key) {
    for (int64_t i = 0; i < dict->size; i++) {
        if (dict->entries[i].key && strcmp(dict->entries[i].key, key) == 0) {
            return dict->entries[i].value;
        }
    }
    return NULL;
}

// Insert or update a key
void stacker_dict_set(StackerDict* dict, const char* key, const char* value) {
    for (int64_t i = 0; i < dict->size; i++) {
        if (dict->entries[i].key && strcmp(dict->entries[i].key, key) == 0) {
            free(dict->entries[i].value);
            dict->entries[i].value = strdup(value);
            return;
        }
    }
    // If not found, insert into empty slot
    for (int64_t i = 0; i < dict->size; i++) {
        if (dict->entries[i].key == NULL) {
            dict->entries[i].key = strdup(key);
            dict->entries[i].value = strdup(value);
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
    int64_t (*thunk)(void);
} StackerLazy;

// Create a lazy thunk
StackerLazy* stacker_lazy_new(int64_t (*fn)(void)) {
    StackerLazy* lazy = (StackerLazy*) malloc(sizeof(StackerLazy));
    lazy->evaluated = 0;
    lazy->cached_value = 0;
    lazy->thunk = fn;
    return lazy;
}

// Force evaluation
int64_t stacker_lazy_force(StackerLazy* lazy) {
    if (!lazy->evaluated) {
        lazy->cached_value = lazy->thunk();
        lazy->evaluated = 1;
    }
    return lazy->cached_value;
}

// Free lazy value
void stacker_lazy_free(StackerLazy* lazy) {
    free(lazy);
}
