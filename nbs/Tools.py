from osgeo import gdal
import numpy as np
from datetime import datetime
# from scipy import stats
import matplotlib.pyplot as plt
from osgeo import ogr, osr
import random


def timestamp(dt):
    epoch = datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000.0


# def linear_trend(array_x, array_y):
#     n = array_x.shape[0]
#     linear_reg = [array_y[i, :, 0]*stats.linregress(array_x[i, ...])[0]+stats.linregress(array_x[i, ...])[1] for i in range(0, n)]
#     return np.array(linear_reg)


def RGBplot(image, bands):
    img = np.dstack((image[bands.index('B4'), ...]/3000, image[bands.index('B3'), ...]/3000,
                     image[bands.index('B2'), ...]/3000))
    img = np.clip(img, 0, 1)
    f = plt.figure()
    plt.imshow(np.array(img))
    plt.axis('off')
    plt.show()


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


def getPixelValues(shapes, tiffGdal, fieldname, data_img):
    """ general function to intersect polygons/multipolygons with a group of multiband rasters
        IMPORTANT
        polygons and raster must have the same coordinate system!!!
        the bands of a raster must have the same data type
        feature falling partially or totally outside the raster will not be considered
        when passing the subset as a dictionary be sure to use the same rastermask options used for the subset source

    :param shapes: polygons/multipolygons shapefile
    :param folderpath: folder with multiband rasters
    :param fieldname: vector fieldname that contains the labelvalue
    :param images: a list of images to process (not the absolute path)
    :param rastermask: raster where value 0 is the mask
    :param  subset: integer or dictionary
                    - integer percentage (> 0; <100) deciding how much of each polygon you want to consider
                    - a dictionary { polygonID: numpy.ndarray} where the numpy.ndarray is used to apply fancy index
                    to filter the polygon with ID == polygonID
    :param  returnsubset: bool, if true a subset datastructure { polygonID: numpy.ndarray} is returned
    :return: 1) a 2d numpy array,
                each row contains the polygonID column, the unique id column, the pixel
                values for each raster band plus a column with the label:
                the array shape is (numberpixels, numberofrasters*nbands + 3)

                if mask the max numberpixels  per polygon may decrease
                if subset the numberpixels will decrease
             2) a set with the unique labels
             3) a list with column names
             4) if returnsubset is True will return the subset datastructure { polygonID: numpy.ndarray}
    """

    raster = None
    shp = None
    lyr = None
    target_ds = None
    outDataSet = None
    outLayer = None
    band = None
    pixelmask = None
    outdata = []

    m = np.zeros((1157, 1553))
    where = np.zeros((1157, 1553))
    idx = []

    try:

        shp = ogr.Open(shapes)
        lyr = shp.GetLayer()

        sourceSR = lyr.GetSpatialRef()

        # get number of features; get number of bands
        featureCount = lyr.GetFeatureCount()

        # iterate features and extract unique labels
        classValues = []
        for feature in lyr:
            classValues.append(feature.GetField(fieldname))
        # get the classes unique values
        uniqueLabels = set(classValues)
        # reset the iterator
        lyr.ResetReading()
        # get the content of the images directory
        ########imgs= os.listdir(folderpath)

        label = None
        columnNames = []
        labels = []  #this will store all the labels

        # iterate all the files and keep only the ones with the correct extension
        ##### filter content, we want files with the correct extension
        #####if os.path.isfile(folderpath+'/'+i) and (os.path.splitext(folderpath+'/'+i)[-1] in inimgfrmt) :

        # open raster data
        # raster = gdal.Open(folderpath+'/'+i,gdalconst.GA_ReadOnly)
        raster = tiffGdal
        nbands = data_img.shape[2]

        # we need to get the raster datatype for later use (assumption:every band has the same data type)
        band = raster.GetRasterBand(1)
        raster_data_type = band.DataType

        # Get raster georeference info
        width = raster.RasterXSize
        height = raster.RasterYSize

        transform = raster.GetGeoTransform()
        xOrigin = minx = transform[0] # Top left coordinates of the starting raster
        yOrigin = maxy =  transform[3]
        miny = transform[3] + width*transform[4] + height*transform[5]
        maxx = transform[0] + width*transform[1] + height*transform[2]
        pixelWidth = transform[1]
        pixelHeight = transform[5]

        numfeature = 0

        # keep trak of the number of ids, necessary to assign id to subsequent polygons
        idcounter = 1

        # reset the iterator
        lyr.ResetReading()

        intermediatedata = []

        for feat in lyr:

            numfeature += 1
            print("working on feature %d of %d" % (numfeature, featureCount))

            #get the label and the polygon ID
            label = feat.GetField(fieldname)
            polygonID = feat.GetFID() + 1  #I add one to avoid the first polygonID==0

            #  Get extent of feature
            geom = feat.GetGeometryRef()
            if geom.GetGeometryName() == "MULTIPOLYGON":
                count = 0
                pointsX = []; pointsY = []
                for polygon in geom:
                    geomInner = geom.GetGeometryRef(count)
                    ring = geomInner.GetGeometryRef(0)
                    numpoints = ring.GetPointCount()
                    for p in range(numpoints):
                        lon, lat, z = ring.GetPoint(p)
                        pointsX.append(lon)
                        pointsY.append(lat)
                    count += 1
            elif geom.GetGeometryName() == "POLYGON":
                ring = geom.GetGeometryRef(0)
                numpoints = ring.GetPointCount()
                pointsX = []
                pointsY = []
                for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)

            else:
                raise Exception("ERROR: Geometry needs to be either Polygon or Multipolygon")

            xmin = min(pointsX)
            xmax = max(pointsX)
            ymin = min(pointsY)
            ymax = max(pointsY)

            #check if this feature is completely inside the raster, if not skip it
            if any([xmin < minx, xmax > maxx, ymin < miny, ymax > maxy]):
                print('feature with id = %d is falling outside the raster and will not be considered'%feat.GetFID())
                continue

            # Specify offset and rows and columns to read
            # Offset of the little raster, created to wrap each polygon feature
            xoff = int((xmin - xOrigin)/pixelWidth)
            yoff = int((yOrigin - ymax)/pixelWidth)
            # print("Offset of little raster: ", xoff, yoff)

            # Number of rows and columns of the little raster
            xcount = int((xmax - xmin)/pixelWidth)+1
            ycount = int((ymax - ymin)/pixelWidth)+1
            # print("Rows and cols of little raster: ", xcount, ycount)


            # Create memory target multiband raster, with the same nbands and datatype as the input raster
            target_ds = gdal.GetDriverByName("MEM").Create("", xcount, ycount, nbands, raster_data_type)
            target_ds.SetGeoTransform((
                xmin, pixelWidth, 0,
                ymax, 0, pixelHeight,
            ))

            # Create for target raster the same projection as for the value raster
            raster_srs = osr.SpatialReference()
            raster_srs.ImportFromWkt(raster.GetProjectionRef())
            target_ds.SetProjection(raster_srs.ExportToWkt())

            #create in memory vector layer that contains the feature
            drv = ogr.GetDriverByName("ESRI Shapefile")
            outDataSet = drv.CreateDataSource("/vsimem/memory.shp")
            outLayer = outDataSet.CreateLayer("memoryshp", srs=sourceSR, geom_type=lyr.GetGeomType())

            # set the output layer's feature definition
            outLayerDefn = lyr.GetLayerDefn()
            # create a new feature
            outFeature = ogr.Feature(outLayerDefn)
            # set the geometry and attribute
            outFeature.SetGeometry(geom)
            # add the feature to the shapefile
            outLayer.CreateFeature(outFeature)

            # Rasterize zone polygon to raster
            # outputraster, list of bands to update, input layer, list of values to burn
            gdal.RasterizeLayer(target_ds, list(range(1, nbands+1)), outLayer, burn_values=[label]*nbands)

            # Read rasters as arrays
            dataraster = np.moveaxis(data_img[yoff:yoff+ycount, xoff:xoff+xcount, :], 2, 0)
            # raster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
            datamask = target_ds.ReadAsArray(0, 0, xcount, ycount).astype(np.float)

            #extract the data for each band
            data = []
            for j in range(nbands):
                data.append(dataraster[j][datamask[j] > 0])

            #define label data for this polygon
            label = (np.zeros(data[0].shape[0]) + label).reshape(data[0].shape[0], 1)
            polygonIDarray = (np.zeros(data[0].shape[0]) + polygonID).reshape(data[0].shape[0], 1)
            # fill in the list with all the labels, this will be the last column in the final output

            id = np.arange(idcounter,(data[0].shape[0]) + idcounter).reshape(data[0].shape[0], 1) #+1 is there to avoid first polygon different from 0

            # update the starting id for the next polygon
            idcounter += data[0].shape[0]
            vstackdata = np.vstack(data).T


            #if subset we need to define the correct fancy indexing
            intermediatedata.append(np.hstack((polygonIDarray, id,  vstackdata)))
            labels.append(label)

            #give control back to c++ to free memory
            target_ds = None
            outLayer = None
            outDataSet = None

        #########END for feat in lyr


        #store the field names
        columnNames += ["polyID\t", "id\t"]
        for k in range(nbands):
            columnNames.append("_b" + str(k+1)+"\t")

        # stack vertically the output of each feature class
        outdata.append(np.vstack(intermediatedata))

        # stack horizontally
        outdata = np.hstack(outdata)
        # finally append the lables at the end
        outdata = np.hstack((outdata, np.vstack(labels)))

        columnNames.append("label")

        return outdata, uniqueLabels, columnNames

    finally:

        # give control back to c++ to free memory
        if raster:
            raster = None
        if pixelmask:
            pixelmask = None
        if shp:
            shp = None
        if lyr:
            lyr = None
        if target_ds:
            target_ds = None
        if outLayer:
            outLayer = None
        if outDataSet:
            outDataSet = None
        if band:
            band = None


def ind_VfoldCross(data, selec):
    random.seed(30)

    cls = np.unique(data)
    arr_train = []

    for i in cls:
        #get the indexes for each
        ind = np.where(data == i)

        if len(ind[0]) < selec:
            sel = random.sample(range(len(ind[0])), len(ind[0]))
            sel = [sel[i] for i in range(int(np.round(2*len(ind[0])/3)))]
            arr_train.extend(ind[0][sel])
        else:
            sel = random.sample(range(len(ind[0])), selec)
            arr_train.extend(ind[0][sel])

    return arr_train


def VegetationIndex(y1, bands, opt):
    print('vegetation indices')
    y_8bits = [((y1[i, :, :] - np.min(y1[i, :, :])) / (np.max(y1[i, :, :]) - np.min(y1[i, :, :]))) * 255 for i in
               range(0, y1.shape[0])]
    # # Vegetation indices:
    # NDVI veg_index <- (y[,nir]-y[,red])/(y[,nir]+y[,red])
    ndvi = (y_8bits[bands.index('B8')] - y_8bits[bands.index('B4')]) / \
           (y_8bits[bands.index('B8')] + y_8bits[bands.index('B4')])
    if opt == 1:
        f = plt.figure()
        plt.imshow(ndvi, vmin=-1, vmax=1)

    vi = np.atleast_3d(ndvi)

    # NDWI <- (y[,green]-y[,nir])/(y[,green]+y[,nir])
    ndwi = (y_8bits[bands.index('B3')] - y_8bits[bands.index('B8')]) /\
           (y_8bits[bands.index('B3')] + y_8bits[bands.index('B8')])

    if opt == 1:
        f = plt.figure()
        plt.imshow(ndwi, vmin=-1, vmax=1)
    vi = np.concatenate((vi, np.atleast_3d(ndwi)), axis=2)

    # NDWI <- (y[,NIR]-y[,SWIR])/(y[,NIR]+y[,SWIR])
    ndmi = (y_8bits[bands.index('B8')] - y_8bits[bands.index('B11')]) /\
           (y_8bits[bands.index('B8')] + y_8bits[bands.index('B11')])
    if opt == 1:
        f = plt.figure()
        plt.imshow(ndmi, vmin=-1, vmax=1)

    vi = np.concatenate((vi, np.atleast_3d(ndmi)), axis=2)

    # EVI veg_index <- 2.5*(y[,nir]-y[,red])/(y[,nir]+6*y[,red]-7.5*y[,blue]+1)
    evi = 2.5 * (y_8bits[bands.index('B8')] - y_8bits[bands.index('B4')]) / \
        (y_8bits[bands.index('B8')] + 6 * y_8bits[bands.index('B4')] - 7.5 *
         y_8bits[bands.index('B2')] + 1)
    evi = ((evi - np.percentile(evi, 5)) / (np.percentile(evi, 95) - np.percentile(evi, 5)))
    # evi[evi < -1] = -1
    # evi[evi > 1] = 1
    if opt == 1:
        f = plt.figure()
        plt.imshow(evi, vmin=-1, vmax=1)
    # evi = 2.5 * (y[8, :, :] - y[3, :, :]) / (y[8, :, :] + 6 * y[3, :, :] - 7.5 * y[1, :, :] + 1)
    vi = np.concatenate((vi, np.atleast_3d(evi)), axis=2)

    # TCARI veg_index <- 3*((y[,red_edge]-y[,red])-0.2*(y[,red_edge]-y[,green])*(y[,red_edge]/y[,red]))
    tcari = 3 * (y_8bits[bands.index('B7')] - y_8bits[bands.index('B4')]) - \
        0.2 * (y_8bits[bands.index('B7')] - y_8bits[bands.index('B3')]) * (
                y_8bits[bands.index('B7')] / (y_8bits[bands.index('B4')] + np.finfo(float).eps))
    tcari = ((tcari - np.percentile(tcari, 5)) / (np.percentile(tcari, 95) - np.percentile(tcari, 5)))
    if opt == 1:
        f = plt.figure()
        plt.imshow(tcari, vmin=-1, vmax=1)
    vi = np.concatenate((vi, np.atleast_3d(tcari)), axis=2)

    # SAVI veg_index <- (1+L)*(y[,nir] - y[,red])/(y[,nir] + y[,red] + L) L=0.5
    # savi = 1.5 * (y[7, :, :] - y[3, :, :]) / (y[7, :, :] + y[3, :, :] + 0.5)
    savi = 1.5 * (y_8bits[bands.index('B8')] - y_8bits[bands.index('B4')]) / \
        (y_8bits[bands.index('B8')] + y_8bits[bands.index('B4')] + 0.5)
    savi = ((savi - np.percentile(savi, 5)) / (np.percentile(savi, 95) - np.percentile(savi, 5)))

    if opt == 1:
        f = plt.figure()
        plt.imshow(savi, vmin=-1, vmax=1)

    vi = np.concatenate((vi, np.atleast_3d(savi)), axis=2)

    # # kernel NDVI:
    pr = [(y_8bits[i] - np.min(y_8bits[i])) / (np.max(y_8bits[i]) - np.min(y_8bits[i])) for i in range(len(y_8bits))]
    sigma = np.nanmedian(np.power(pr[bands.index('B8')] - pr[bands.index('B4')], 2))
    ker = np.exp(np.power(pr[bands.index('B8')] - pr[bands.index('B4')], 2)/(-2*(np.power(sigma, 2))))
    kndvi = (1-ker) / (1+ker)
    # vi = np.concatenate((vi, np.atleast_3d(kndvi)), axis=2)

    # MSAVI veg_index <- 0.5*(2*y[,nir]+1-sqrt(((2*y[,nir]+1)^2)-8*(y[,nir]-y[,red])))
    # msavi2 = 0.5 * (2 * y[7, :, :] + 1 - np.sqrt(((2 * y[7, :, :] + 1) ** 2)
    #                                              - 8 * (y[7, :, :] - y[3, :, :])))
    msavi2 = 0.5 * (2 * y_8bits[bands.index('B8')] + 1 -
                    np.sqrt(((2 * y_8bits[bands.index('B8')] + 1) ** 2) -
                            8 * (y_8bits[bands.index('B8')] - y_8bits[bands.index('B4')])))
    msavi2 = ((msavi2 - np.percentile(msavi2, 5)) / (np.percentile(msavi2, 95) - np.percentile(msavi2, 5)))
    if opt == 1:
        f = plt.figure()
        plt.imshow(msavi2, vmin=-1, vmax=1)
    vi = np.concatenate((vi, np.atleast_3d(msavi2)), axis=2)

    # # GLI veg_index <- (2*y[,green]] - y[,red] - y[,blue])/(2*y[,green] + y[,red] + y[,blue])
    # # gli = (2 * y[2, :, :] - y[3, :, :] - y[1, :, :]) / (2 * y[2, :, :] + y[3, :, :] + y[1, :, :])
    # gli = (2 * y_8bits[bands.index('B3')] - y_8bits[bands.index('B4')] -
    #        y_8bits[bands.index('B2')]) / \
    #       (2 * y_8bits[bands.index('B3')] + y_8bits[bands.index('B4')] +
    #        y_8bits[bands.index('B2')])
    # gli = ((gli - np.percentile(gli, 5)) / (np.percentile(gli, 95) - np.percentile(gli, 5)))
    # if opt == 1:
    #     f = plt.figure()
    #     plt.imshow(gli, vmin=-1, vmax=1)
    #
    # vi = np.concatenate((vi, np.atleast_3d(gli)), axis=2)
    return vi


def diff_emma(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def maskGeneration(vector_layer, raster_layer, target_layer):
# vector_layer = r"C:\Users\Emma\Desktop\ACube4Floods\code\GT_data\29_10_18_Gail.shp"
# raster_layer = r"C:\Users\Emma\Desktop\ACube4Floods\code\data\load_Carinthia\clip\Carinthia_clean_Image_20181117.tif"
# target_layer = r"C:\Users\Emma\Desktop\ACube4Floods\code\mask.tif"

    # open the raster layer and get its relevant properties
    raster_ds = gdal.Open(raster_layer, gdal.GA_ReadOnly)
    xSize = raster_ds.RasterXSize
    ySize = raster_ds.RasterYSize
    geotransform = raster_ds.GetGeoTransform()
    projection = raster_ds.GetProjection()

    # create the target layer (1 band)
    driver = gdal.GetDriverByName('GTiff')
    target_ds = driver.Create(target_layer, xSize, ySize, bands = 1, eType = gdal.GDT_Byte, options = ["COMPRESS=DEFLATE"])
    target_ds.SetGeoTransform(geotransform)
    target_ds.SetProjection(projection)

    # rasterize the vector layer into the target one
    ds = gdal.Rasterize(target_ds, vector_layer, burnValues=[1])

    target_ds = None
    return ds


def pls(X, Y):
    Cxy = X.T.dot(Y)
    U_pls, S, V = np.linalg.svd(Cxy, full_matrices=False)
    ind = np.argsort(S)[::-1]
    S = S[ind]
    U_pls = U_pls[:, ind]
    return S, U_pls, U_pls.shape[1]


def predict(X_train, U_pls, X_test, Yb):
    XtrainProj = X_train.dot(U_pls)
    XtestProj = X_test.dot(U_pls)

    XtrainProj1 = np.concatenate((XtrainProj, np.ones((XtrainProj.shape[0], 1))), axis=1)
    W = np.linalg.pinv(XtrainProj1).dot(Yb)
    Ypred = np.add(XtestProj.dot(W[:-1, :]), np.repeat(W[:-1, :], [XtestProj.shape[0]-1, 1], axis=0))
    Ypred = np.argmax(Ypred, axis=1)
    return Ypred

