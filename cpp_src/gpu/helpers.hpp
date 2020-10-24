#pragma once

#define NUMERIC_TYPES(MACRO) MACRO(float); \
                             MACRO(int); \
                             MACRO(double); \
                             MACRO(char); \
                             MACRO(unsigned int); \
                             MACRO(unsigned char); \
                             MACRO(unsigned short); \
                             MACRO(short);

#define REAL_TYPES(MACRO)    MACRO(float); \
                             MACRO(double);

