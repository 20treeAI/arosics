# -*- coding: utf-8 -*-
__author__='Daniel Scheffler'

import warnings
import sys

# custom
import numpy  as np
from geopandas import GeoDataFrame

try:
    import gdal
    import osr
    import ogr
except ImportError:
    from osgeo import gdal
    from osgeo import osr
    from osgeo import ogr

# internal modules
from py_tools_ds.ptds                      import GeoArray
from py_tools_ds.ptds.geo.coord_calc       import calc_FullDataset_corner_positions
from py_tools_ds.ptds.geo.coord_trafo      import pixelToMapYX, imYX2mapYX
from py_tools_ds.ptds                      import GeoArray





def angle_to_north(XY):
    """Calculates the angle between the lines [origin:[0,0],north:[0,1]] and
     [origin:[0,0],pointXY:[X,Y]] in clockwise direction. Returns values between 0 and 360 degrees.
     """
    XY    = np.array(XY)
    XYarr = XY if len(XY.shape)==2 else XY.reshape((1,2))
    return np.abs(np.degrees(np.arctan2(XYarr[:,1],XYarr[:,0])-np.pi/2)%360)


def get_true_corner_mapXY(fPath_or_geoarray, bandNr=1, noDataVal=None, mp=1, v=0, q=0):
    geoArr    = GeoArray(fPath_or_geoarray) if not isinstance(fPath_or_geoarray,GeoArray) else fPath_or_geoarray

    rows,cols = geoArr.shape[:2]
    gt, prj   = geoArr.geotransform, geoArr.projection
    assert gt and prj, 'GeoTransform an projection must be given for calculation of LonLat corner coordinates.'

    mask_1bit = np.zeros((rows,cols),dtype='uint8') # zeros -> image area later overwritten by ones

    if noDataVal is None:
        mask_1bit[:,:] = 1
    elif noDataVal=='unclear':
        warnings.warn("No data value could not be automatically detected. Thus the matching window used for shift "
              "calulation had to be centered in the middle of the overlap center without respecting no data values. "
              "To avoid this provide the correct no data values for reference and shift image via '-nodata'")
        mask_1bit[:,:] = 1
    else:
        band_data = geoArr[bandNr-1] # TODO implement gdal_ReadAsArray_mp (reading in multiprocessing)
        mask_1bit[band_data!=noDataVal] = 1

    if v: print('detected no data value',noDataVal)

    try:
        corner_coords_YX = calc_FullDataset_corner_positions(mask_1bit, assert_four_corners=False, algorithm='shapely')
    except Exception:
        if v:
            warnings.warn("\nCalculation of corner coordinates failed within algorithm 'shapely' (Exception: %s)."
                          " Using algorithm 'numpy' instead." %sys.exc_info()[1])
        corner_coords_YX = calc_FullDataset_corner_positions(mask_1bit, assert_four_corners=False, algorithm='numpy')

    if len(corner_coords_YX)==4: # this avoids shapely self intersection
        corner_coords_YX = list(np.array(corner_coords_YX)[[0,1,3,2]]) # UL, UR, LL, LR => UL, UR, LR, LL

    # check if enough unique coordinates have been found
    if not len(GeoDataFrame(corner_coords_YX).drop_duplicates().values)>=3:
        if not q:
            warnings.warn('\nThe algorithm for automatically detecting the actual image coordinates did not find '
                          'enough unique corners. Using outer image corner coordinates instead.')
        corner_coords_YX = ((0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1))

    # check if all points are unique
    #all_coords_are_unique = len([UL, UR, LL, LR]) == len(GeoDataFrame([UL, UR, LL, LR]).drop_duplicates().values)
    #UL, UR, LL, LR = (UL, UR, LL, LR) if all_coords_are_unique else ((0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1))

    get_mapYX = lambda YX: pixelToMapYX(list(reversed(YX)),geotransform=geoArr.geotransform,projection=prj)[0]
    corner_pos_XY = [list(reversed(i)) for i in [get_mapYX(YX) for YX in corner_coords_YX]]
    return corner_pos_XY


def get_subset_GeoTransform(gt_fullArr,subset_box_imYX):
    gt_subset = list(gt_fullArr[:]) # copy
    gt_subset[3],gt_subset[0] = imYX2mapYX(subset_box_imYX[0],gt_fullArr)
    return gt_subset


def get_gdalReadInputs_from_boxImYX(boxImYX):
    """Returns row_start,col_start,rows_count,cols_count and assumes boxImYX as [UL_YX,UR_YX,LR_YX,LL_YX)"""
    rS, cS = boxImYX[0]
    clip_sz_x = abs(boxImYX[1][1]-boxImYX[0][1]) # URx-ULx
    clip_sz_y = abs(boxImYX[0][0]-boxImYX[3][0]) # ULy-LLy
    return cS, rS, clip_sz_x,clip_sz_y


def get_GeoArrayPosition_from_boxImYX(boxImYX):
    """Returns row_start,row_end,col_start,col_end and assumes boxImYX as [UL_YX,UR_YX,LR_YX,LL_YX)"""
    rS, cS = boxImYX[0] # UL
    rE, cE = boxImYX[2] # LR
    return rS, rE, cS, cE


def find_noDataVal(pathIm_or_GeoArray,bandIdx=0,sz=3):
    """tries to derive no data value from homogenious corner pixels within 3x3 windows (by default)
    :param pathIm_or_GeoArray:
    :param bandIdx:
    :param sz: window size in which corner pixels are analysed
    """
    geoArr       = pathIm_or_GeoArray if isinstance(pathIm_or_GeoArray, GeoArray) else GeoArray(pathIm_or_GeoArray)
    get_mean_std = lambda corner_subset: {'mean':np.mean(corner_subset), 'std':np.std(corner_subset)}
    UL           = get_mean_std(geoArr[0:sz,0:sz,bandIdx])
    UR           = get_mean_std(geoArr[0:sz,-sz:,bandIdx])
    LR           = get_mean_std(geoArr[-sz:,-sz:,bandIdx])
    LL           = get_mean_std(geoArr[-sz:,0:sz,bandIdx])
    possVals     = [i['mean'] for i in [UL,UR,LR,LL] if i['std']==0]
    # possVals==[]: all corners are filled with data; np.std(possVals)==0: noDataVal clearly identified
    return None if possVals==[] else possVals[0] if np.std(possVals)==0 else 'unclear'