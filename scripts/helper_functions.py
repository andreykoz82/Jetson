from ctypes.util import find_library
import os
import ctypes


def get_abs_path_of_library(library):
    abs_path = None
    lib_name = find_library(library)
    if os.path.exists(lib_name):
        abs_path = os.path.abspath(lib_name)
        return abs_path
    libdl = ctypes.CDLL(lib_name)
    if not libdl:
        return abs_path
    try:
        dlinfo = libdl.dlinfos
    except AttributeError as err:
        abs_path = str(err).split(":")[0]
    return abs_path


def c_wrapper():
    return """
    typedef signed char             l_int8;
typedef unsigned char           l_uint8;
typedef short                   l_int16;
typedef unsigned short          l_uint16;
typedef int                     l_int32;
typedef unsigned int            l_uint32;
typedef float                   l_float32;
typedef double                  l_float64;
typedef long long               l_int64;
typedef unsigned long long      l_uint64;
typedef int l_ok; /*!< return type 0 if OK, 1 on error */


struct Pix;
typedef struct Pix PIX;
typedef enum lept_img_format {
    IFF_UNKNOWN        = 0,
    IFF_BMP            = 1,
    IFF_JFIF_JPEG      = 2,
    IFF_PNG            = 3,
    IFF_TIFF           = 4,
    IFF_TIFF_PACKBITS  = 5,
    IFF_TIFF_RLE       = 6,
    IFF_TIFF_G3        = 7,
    IFF_TIFF_G4        = 8,
    IFF_TIFF_LZW       = 9,
    IFF_TIFF_ZIP       = 10,
    IFF_PNM            = 11,
    IFF_PS             = 12,
    IFF_GIF            = 13,
    IFF_JP2            = 14,
    IFF_WEBP           = 15,
    IFF_LPDF           = 16,
    IFF_TIFF_JPEG      = 17,
    IFF_DEFAULT        = 18,
    IFF_SPIX           = 19
};

char * getLeptonicaVersion (  );
PIX * pixRead ( const char *filename );
PIX * pixCreate ( int width, int height, int depth );
PIX * pixEndianByteSwapNew(PIX  *pixs);
l_int32 pixSetData ( PIX *pix, l_uint32 *data );
l_ok pixSetPixel ( PIX *pix, l_int32 x, l_int32 y, l_uint32 val );
l_ok pixWrite ( const char *fname, PIX *pix, l_int32 format );
l_int32 pixFindSkew ( PIX *pixs, l_float32 *pangle, l_float32 *pconf );
PIX * pixDeskew ( PIX *pixs, l_int32 redsearch );
void pixDestroy ( PIX **ppix );
l_ok pixGetResolution ( const PIX *pix, l_int32 *pxres, l_int32 *pyres );
l_ok pixSetResolution ( PIX *pix, l_int32 xres, l_int32 yres );
l_int32 pixGetWidth ( const PIX *pix );

typedef struct TessBaseAPI TessBaseAPI;
typedef struct ETEXT_DESC ETEXT_DESC;
typedef struct TessPageIterator TessPageIterator;
typedef struct TessResultIterator TessResultIterator;
typedef int BOOL;

typedef enum TessOcrEngineMode  {
    OEM_TESSERACT_ONLY          = 0,
    OEM_LSTM_ONLY               = 1,
    OEM_TESSERACT_LSTM_COMBINED = 2,
    OEM_DEFAULT                 = 3} TessOcrEngineMode;

typedef enum TessPageSegMode {
    PSM_OSD_ONLY               =  0,
    PSM_AUTO_OSD               =  1,
    PSM_AUTO_ONLY              =  2,
    PSM_AUTO                   =  3,
    PSM_SINGLE_COLUMN          =  4,
    PSM_SINGLE_BLOCK_VERT_TEXT =  5,
    PSM_SINGLE_BLOCK           =  6,
    PSM_SINGLE_LINE            =  7,
    PSM_SINGLE_WORD            =  8,
    PSM_CIRCLE_WORD            =  9,
    PSM_SINGLE_CHAR            = 10,
    PSM_SPARSE_TEXT            = 11,
    PSM_SPARSE_TEXT_OSD        = 12,
    PSM_COUNT                  = 13} TessPageSegMode;

typedef enum TessPageIteratorLevel {
     RIL_BLOCK    = 0,
     RIL_PARA     = 1,
     RIL_TEXTLINE = 2,
     RIL_WORD     = 3,
     RIL_SYMBOL    = 4} TessPageIteratorLevel;    

TessPageIterator* TessBaseAPIAnalyseLayout(TessBaseAPI* handle);
TessPageIterator* TessResultIteratorGetPageIterator(TessResultIterator* handle);

BOOL TessPageIteratorNext(TessPageIterator* handle, TessPageIteratorLevel level);
BOOL TessPageIteratorBoundingBox(const TessPageIterator* handle, TessPageIteratorLevel level,
                                 int* left, int* top, int* right, int* bottom);

const char* TessVersion();

TessBaseAPI* TessBaseAPICreate();
int    TessBaseAPIInit3(TessBaseAPI* handle, const char* datapath, const char* language);
int    TessBaseAPIInit2(TessBaseAPI* handle, const char* datapath, const char* language, TessOcrEngineMode oem);
void   TessBaseAPISetPageSegMode(TessBaseAPI* handle, TessPageSegMode mode);
void   TessBaseAPISetImage(TessBaseAPI* handle,
                           const unsigned char* imagedata, int width, int height,
                           int bytes_per_pixel, int bytes_per_line);
void   TessBaseAPISetImage2(TessBaseAPI* handle, struct Pix* pix);

BOOL   TessBaseAPISetVariable(TessBaseAPI* handle, const char* name, const char* value);
BOOL   TessBaseAPIDetectOrientationScript(TessBaseAPI* handle, char** best_script_name, 
                                                            int* best_orientation_deg, float* script_confidence, 
                                                            float* orientation_confidence);
int TessBaseAPIRecognize(TessBaseAPI* handle, ETEXT_DESC* monitor);
TessResultIterator* TessBaseAPIGetIterator(TessBaseAPI* handle);
BOOL   TessResultIteratorNext(TessResultIterator* handle, TessPageIteratorLevel level);
char*  TessResultIteratorGetUTF8Text(const TessResultIterator* handle, TessPageIteratorLevel level);
float  TessResultIteratorConfidence(const TessResultIterator* handle, TessPageIteratorLevel level);
char*  TessBaseAPIGetUTF8Text(TessBaseAPI* handle);
const char*  TessResultIteratorWordFontAttributes(const TessResultIterator* handle, BOOL* is_bold, BOOL* is_italic,
                                                              BOOL* is_underlined, BOOL* is_monospace, BOOL* is_serif,
                                                              BOOL* is_smallcaps, int* pointsize, int* font_id);
void   TessBaseAPIEnd(TessBaseAPI* handle);
void   TessBaseAPIDelete(TessBaseAPI* handle);
    """
