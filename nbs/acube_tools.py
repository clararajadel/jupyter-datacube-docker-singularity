import gdal
import numpy as np
# from scipy import stats
import matplotlib.pyplot as plt
from osgeo import ogr, osr
import random
from datetime import timedelta, date
import datacube
acube = datacube.Datacube(app='boku', env='acube')
import math
import datetime
from datetime import timedelta, date, datetime
from datacube.utils.cog import write_cog
from pylab import *
import os
import matplotlib.colors as clr
import itertools
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
import joblib

def get_long_lat (tif):
    ds = gdal.Open(tif)
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5] 
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]
    longitude = (minx, maxx)
    latitude = (miny, maxy)
    return longitude, latitude


def query_data (latitude, longitude, date):
    query = {
        'product': 'B_Sentinel_2',
        'output_crs': 'EPSG:32633',
        'resolution': (-10, 10),
        'lon': longitude,
        'lat': latitude,
        'time': date,
        'measurements': ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'],
        'cloud_cover_percentage': (0.0, 50.0)
    }
    data = acube.load(**query)
    data_array = np.array(data.to_array(), dtype='float')
    for dataset in acube.find_datasets_lazy(**query):
        geo = dataset.metadata_doc['extent']['coord']['ll']
    return data_array, geo


def image_selection_byCloud(fdate, latitude, longitude, cpercen, mode):
    '''
    fdate = tuple(YYYY, MM, DD)
    latitude = tuple(xx, yy)
    longitude = tuple(xx, yy)
    cpercen = str
    mode = str('before') or str('after')
    '''

    # date range
    sdate = fdate  # start date: flood date
    if mode == 'before':
        edate = date(2015, 7, 1)  # end date: first sentinel 2 image
        delta = sdate - edate  # as timedelta
        time = (str(edate), str(sdate))  # time range for query
    elif mode == 'after':
        now = datetime.datetime.now()
        edate = date(now.year, now.month, now.day)  # end date: today
        delta = edate - sdate  # as timedelta
        time = (str(sdate), str(edate))  # time range for query
    else:
        print('No valid mode')

    # list available dates in acube: they are not ordered
    c_product = 'CLOUDMASK_Sentinel_2'
    c_query = {
        'lat': latitude,
        'lon': longitude,
        'time': time
    }
    acube_dates = []
    for dataset in acube.find_datasets_lazy(product=c_product, **c_query):
        acube_dates.append((dataset.center_time).strftime("%Y-%m-%d"))

    # for loop per available days: see cloud coverage
    for i in range(delta.days + 1):
        if mode == 'before':
            day = (sdate - timedelta(days=i)).strftime("%Y-%m-%d")
        elif mode == 'after':
            day = (sdate + timedelta(days=i)).strftime("%Y-%m-%d")
        print('----------------------------------')
        print('New Date:', day)
        # query cloud data
        if day in acube_dates:
            print(f'day {day} is in acube_dates')
            # print('Cloud data querying...')
            cloud_query = {
                'product': 'CLOUDMASK_Sentinel_2',
                'output_crs': 'EPSG:32633',
                'resolution': (-10, 10),
                'lon': longitude,
                'lat': latitude,
                'time': day,
                'measurements': ['band_1'],
            }
            # select image with cloud coverage condition
            cloud = acube.load(**cloud_query)
            # print('Transform query to array...')
            cloud_array = np.array(cloud.to_array())[0]  # there is only one band
            print('Calculating cloud coverage...')
            if len(np.unique(cloud_array[0])) == 1:
                print('No cloud coverage')
                s2_array, s2_geo = query_data(latitude, longitude, day)
                if not np.any(s2_array) == False:
                    print('End')
                    break
                else:
                    print('sentinel 2 band-arrays full of zeros!')
                    continue
            elif len(np.unique(cloud_array[0])) == 2:
                cloud_coverage = np.count_nonzero(cloud_array[0] == 255) / (
                            cloud_array.shape[1] * cloud_array.shape[2]) * 100
                print('Cloud coverage (%):', cloud_coverage)
                if cloud_coverage < cpercen:
                    print(f'Cloud coverage less than {cpercen}%')
                    s2_array, s2_geo = query_data(latitude, longitude, day)
                    if not np.any(s2_array) == False:
                        print('End')
                        break
                    else:
                        print('sentinel 2 band-arrays full of zeros!')
                        continue
                else:
                    print(f'Cloud coverage more than {cpercen}%')
                    continue
            else:
                print('Error in cloud mask')
        else:
            print(f'day {day} is not in acube_dates')
            continue
    # delete temporal dimension (there is only one image)
    s2_array = s2_array.reshape(s2_array.shape[0], s2_array.shape[2], s2_array.shape[3])
    return day, s2_array, cloud_array, s2_geo


def VegetationIndex(y1, filter_col, features):
    # # Vegetation indices:
    y_8bits = [((y1[i, :, :] - np.min(y1[i, :, :])) / (np.max(y1[i, :, :]) - np.min(y1[i, :, :]))) * 255 for i in
               range(0, y1.shape[0])]

    if 'NDVI' in features:
        # NDVI veg_index <- (y[,nir]-y[,red])/(y[,nir]+y[,red])
        ndvi = (y_8bits[filter_col.index('B8')] - y_8bits[filter_col.index('B4')]) / \
               (y_8bits[filter_col.index('B8')] + y_8bits[filter_col.index('B4')] + np.finfo(float).eps)
        ndvi = np.reshape(ndvi, (ndvi.shape[0] * ndvi.shape[1], 1))
        y1 = np.moveaxis(y1, 0, -1)
        y1 = np.reshape(y1, (y1.shape[0] * y1.shape[1], y1.shape[2]))
        y1 = np.concatenate((y1, ndvi), axis=1)

    if 'NDWI' in features:
        # NDWI <- (y[,green]-y[,nir])/(y[,green]+y[,nir])
        ndwi = (y_8bits[filter_col.index('B3')] - y_8bits[filter_col.index('B8')]) / \
               (y_8bits[filter_col.index('B3')] + y_8bits[filter_col.index('B8')] + np.finfo(float).eps)
        ndwi = np.reshape(ndwi, (ndwi.shape[0] * ndwi.shape[1], 1))
        y1 = np.concatenate((y1, ndwi), axis=1)

    if 'NDMI' in features:
        # NDWI <- (y[,NIR]-y[,SWIR])/(y[,NIR]+y[,SWIR])
        ndmi = (y_8bits[filter_col.index('B8')] - y_8bits[filter_col.index('B11')]) / \
               (y_8bits[filter_col.index('B8')] + y_8bits[filter_col.index('B11')] + np.finfo(float).eps)

        ndmi = np.reshape(ndmi, (ndmi.shape[0] * ndmi.shape[1], 1))
        y1 = np.concatenate((y1, ndwi), axis=1)

    if 'EVI' in features:
        # EVI veg_index <- 2.5*(y[,nir]-y[,red])/(y[,nir]+6*y[,red]-7.5*y[,blue]+1)
        evi = 2.5 * (y_8bits[filter_col.index('B8')] - y_8bits[filter_col.index('B4')]) / \
              ((y_8bits[filter_col.index('B8')] + 6 * y_8bits[filter_col.index('B4')] - 7.5 *
                y_8bits[filter_col.index('B2')] + 1) + np.finfo(float).eps)
        evi = np.reshape(evi, (evi.shape[0] * evi.shape[1], 1))
        y1 = np.concatenate((y1, evi), axis=1)

    if 'TCARI' in features:
        # TCARI veg_index <- 3*((y[,red_edge]-y[,red])-0.2*(y[,red_edge]-y[,green])*(y[,red_edge]/y[,red]))
        tcari = 3 * (y_8bits[filter_col.index('B7')] - y_8bits[filter_col.index('B4')]) - \
                0.2 * (y_8bits[filter_col.index('B7')] - y_8bits[filter_col.index('B3')]) * (
                        y_8bits[filter_col.index('B7')] / (y_8bits[filter_col.index('B4')] + np.finfo(float).eps))
        tcari = np.reshape(tcari, (tcari.shape[0] * tcari.shape[1], 1))
        y1 = np.concatenate((y1, tcari), axis=1)

    if 'SAVI' in features:
        # SAVI veg_index <- (1+L)*(y[,nir] - y[,red])/(y[,nir] + y[,red] + L) L=0.5
        # savi = 1.5 * (y[7, :, :] - y[3, :, :]) / (y[7, :, :] + y[3, :, :] + 0.5)
        savi = 1.5 * (y_8bits[filter_col.index('B8')] - y_8bits[filter_col.index('B4')]) / \
               ((y_8bits[filter_col.index('B8')] + y_8bits[filter_col.index('B4')] + 0.5) + np.finfo(float).eps)
        savi = np.reshape(savi, (savi.shape[0] * savi.shape[1], 1))
        y1 = np.concatenate((y1, savi), axis=1)

    if 'KNDVI' in features:
        # # kernel NDVI:
        pr = [(y_8bits[i] - np.min(y_8bits[i])) / (np.max(y_8bits[i]) - np.min(y_8bits[i])) for i in
              range(0, len(y_8bits))]
        sigma_x = 0.15
        # np.nanmedian(np.power(pr[filter_col.index('B8')] - pr[filter_col.index('B4')], 2))
        ker = np.exp(
            np.power(pr[filter_col.index('B8')] - pr[filter_col.index('B4')], 2) / (-2 * (np.power(sigma_x, 2))))
        kndvi = (1 - ker) / (1 + ker)
        kndvi = np.reshape(kndvi, (kndvi.shape[0] * kndvi.shape[1], 1))
        y1 = np.concatenate((y1, kndvi), axis=1)

        if 'MSAVI2' in features:
            # MSAVI veg_index <- 0.5*(2*y[,nir]+1-sqrt(((2*y[,nir]+1)^2)-8*(y[,nir]-y[,red])))
            # msavi2 = 0.5 * (2 * y[7, :, :] + 1 - np.sqrt(((2 * y[7, :, :] + 1) ** 2)
            #                                              - 8 * (y[7, :, :] - y[3, :, :])))
            msavi2 = 0.5 * (2 * y_8bits[filter_col.index('B8')] + 1 -
                            np.sqrt(((2 * y_8bits[filter_col.index('B8')] + 1) ** 2) -
                                    8 * (y_8bits[filter_col.index('B8')] - y_8bits[filter_col.index('B4')])))
        msavi2 = np.reshape(msavi2, (msavi2.shape[0] * msavi2.shape[1], 1))
        y1 = np.concatenate((y1, msavi2), axis=1)
    return y1


def maskGeneration(array, geo_inf, area, subarea, path_gpkg, path_shapefiles, mask_tif):
    # 1° define properties (size, geotransform and crs) from downloaded acube images [insert array]
    # 2° from these properties and defining a path, create the mask tif [insert path/name.tif]
    # 3° import geopackage as shp: it contains flood no flood info, and download all its layers (different areas) [insert path/name.gpkg]
    # 4° combine (rasterize) the mask tif with the shp

    # open raster layer and get its relevant properties
    xSize, ySize = array.shape[1:]
    geotransform = ([geo_inf['lon'], 8.983152858765616e-05, 0.0, geo_inf['lat'], 0.0, -8.983152840909205e-05])
    crs = 'EPSG:32633'

    # create the target layer (mask) (1 band)
    driver = gdal.GetDriverByName('GTiff')
    target_layer = mask_tif
    target_ds = driver.Create(target_layer, xSize, ySize, bands=1, eType=gdal.GDT_Byte, options=["COMPRESS=DEFLATE"])
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(crs)
    
    # import vector layer
    # anyadir funcion que te busque el archivo con el nombre del area porque los anyos cambian --WARNING!!!
    files = sorted(os.listdir(path_gpkg))
    for f in files:
        if f'{area}' in f and f.endswith(('.gpkg')) and os.path.isfile(os.path.join(path_gpkg, f)):
            root_file = path_gpkg+f
    ds = ogr.GetDriverByName('GPKG').Open(root_file, 1)
    source = ogr.Open(root_file, update=False)
    gpkg_layers = [l.GetName() for l in ogr.Open(root_file)]
    drv = ogr.GetDriverByName('ESRI Shapefile')
    masks = []
    for i in gpkg_layers:
        inlyr = source.GetLayer(i)
        outds = drv.CreateDataSource(f'{path_shapefiles}/{area}_pr_' + i + '.shp')
        outlyr = outds.CopyLayer(inlyr, i)
    del inlyr, outlyr, outds

    # rasterize the vector layer into the target one
    files2 = sorted(os.listdir(path_shapefiles))
    for f in files2:
        if area in f and subarea in f and f.endswith(('.shp')) and os.path.isfile(os.path.join(path_gpkg, f)):
            ds = gdal.Rasterize(target_ds, f, burnValues=[1])
            myarray = np.array(ds.GetRasterBand(1).ReadAsArray())
            masks.append(my_array)

    target_ds = None
    return masks


def loadGeoTiff(root_image):
    tifsrc = gdal.Open(root_image)
    in_band = tifsrc.GetRasterBand(1)
    block_xsize, block_ysize = (in_band.XSize, in_band.YSize)
    # read the multiband tile into a 3d numpy array
    image = tifsrc.ReadAsArray(0, 0, block_xsize, block_ysize)
    return image


def ind_VfoldCross(data, selec):
    import random
    random.seed(30)

    cls = np.unique(data)
    arr_train = []

    for i in cls:
        # get the indexes for each
        ind = np.where(data == i)

        if len(ind[0]) < selec:
            sel = random.sample(range(len(ind[0])), len(ind[0]))
            sel = [sel[i] for i in range(int(np.round(2 * len(ind[0]) / 3)))]
            arr_train.extend(ind[0][sel])
        else:
            sel = random.sample(range(len(ind[0])), selec)
            arr_train.extend(ind[0][sel])

    return arr_train


def RandomForestClassification(X_train, Y_train, n_feat, Njobs=None,
                               vfolds=5, Ntree=[100], min_samples_lf=[1],
                               min_samples_sp=[2], save_path):
    import random
    # CROSS-VALIDATION:
    random.seed(999)
    classifiers = []
    cl = np.unique(Y_train)[0]
    print('Cross validation...')
    for ntree in Ntree:
        for mtry in n_feat:
            for lf in min_samples_lf:
                for split in min_samples_sp:
                    scores = []
                    for t in range(0, vfolds):
                        tr_index = ind_VfoldCross(Y_train, np.round(int(len(np.where(Y_train == cl)[0]) / vfolds)))
                        val_index = diff_emma(range(len(Y_train)), tr_index)
                        x_t = X_train[tr_index, :]
                        y_t = Y_train[tr_index]
                        x_val = X_train[val_index, :]
                        y_val = Y_train[val_index]
                        clf = RandomForestClassifier(n_estimators=ntree, max_features=mtry, min_samples_leaf=lf,
                                                     min_samples_split=split, n_jobs=Njobs)
                        clf.fit(x_t, y_t)
                        ypred = clf.predict(x_val)
                        scores.append(accuracy_score(y_val, ypred))
                    # print(scores)
                    classifiers.append([ntree, mtry, lf, split, np.mean(scores)])
                    # print(np.mean(scores))
    classifiers = np.array(classifiers)
    print('CV done!')
    inx = np.where(classifiers == np.amax(classifiers, axis=0)[4])[0]
    BestNtree = classifiers[inx, 0]
    Bestn_feat = classifiers[inx, 1]
    Bestmin_samples_lf = classifiers[inx, 2]
    Bestmin_samples_sp = classifiers[inx, 3]
    print('Training!')

    cl_Final = RandomForestClassifier(n_estimators=int(BestNtree[0]), max_features=int(Bestn_feat[0]),
                                      min_samples_leaf=int(Bestmin_samples_lf[0]),
                                      min_samples_split=int(Bestmin_samples_sp[0]), n_jobs=Njobs)
    cl_Final.fit(X_train, Y_train)
    print('Predicting!')

    # Ypred = cl_Final.predict(X_test)
    # OA = accuracy_score(Y_test, Ypred)
    #
    # kappa = cohen_kappa_score(Y_test, Ypred)
    #
    # CM = confusion_matrix(Y_test, Ypred)
    parameters = [BestNtree, Bestn_feat, Bestmin_samples_lf, Bestmin_samples_sp]
    joblib.dump(rf, f"{save_path}/random_forest.joblib")
    return cl_Final, parameters  # OA, kappa, CM, Ypred


def diff_emma(first, second):
    '''returns "first" but deleting values included in "second"'''

    second = set(second)
    return [item for item in first if item not in second]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)


def rf_save_results(CM, OA, kappa, PEN, SR, target_names, Rootoutput, identifier):
    plt.figure()
    plot_confusion_matrix(CM, classes=target_names, normalize=False, title=identifier,
                          cmap=plt.cm.Blues)
    plt.savefig(Rootoutput + identifier + '.png', format='png', dpi=1000, bbox_inches="tight")
    plt.close('all')

    # save to disk as csv file
    f = Rootoutput + identifier + '.csv'

    headerfile = identifier

    # g = csv.writer(f, dialect='unix')
    with open(f, 'w', newline='') as csvfile:
        g = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        g.writerow(headerfile)
        g.writerow('')
        g.writerow(['CM:'])
        for i in range(CM.shape[0]):
            g.writerow(CM[i, :])
        g.writerow('')
        g.writerow(['OA:', OA])
        g.writerow('')
        g.writerow(['Kappa:', kappa])
        g.writerow('')
        g.writerow(['PEN:', PEN])
        g.writerow('')
        g.writerow(['SR:', SR])


def scale(X):
    cols = []
    descale = []
    for feature in X.T:
        minimum = feature.min(axis=0)
        maximum = feature.max(axis=0)
        col_std = np.divide((feature - minimum), (maximum - minimum))
        cols.append(col_std)
        descale.append((minimum, maximum))
    X_std = np.array(cols)
    return X_std.T, descale


def writeout(array, Rootoutput, identifier, ds_lon, ds_lat):
    print(f'dtype: {array.dtype} to float64')
    array = array.astype('float64')
    cols = array.shape[-2]
    rows = array.shape[-1]
    # acube images are requested with this projection
    crs = 'EPSG:32633'
    # (xmin, xsize, 0, ymin, 0, ysize)
    geotransform = ([ds_lon, 8.983152858765616e-05, 0.0, ds_lat, 0.0, -8.983152840909205e-05])
    driver = gdal.GetDriverByName('GTiff')
    # acube images are uint16
    ds = driver.Create(Rootoutput + identifier + '.tif', rows, cols, 1, gdal.GDT_Float64)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(crs)
    outband = ds.GetRasterBand(1)
    outband.WriteArray(array)
    ds = None
    outband = None
    print('exported')