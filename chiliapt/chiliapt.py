"""
Name:           ChiliAPT.py
Description:    Class for simulate star position on Chili focalplane
输入ifu的ra、dec、PA
输出: 1. IFU和导星在dss星图上的位置
      2. Gaia星表投影在chili焦面上
      3. Gaia星表分别投影在IFU和Guider上的位置 
Author:         Yifei Xiong
Created:        2024-11-9
Modified-History:
2024-11-12 添加了设置导星为中心指向的功能
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MatplotlibPolygon
import shapely
from shapely.geometry import Point, Polygon
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord, Distance
from astropy import units as u
from astropy import wcs as astropy_wcs
from astropy.time import Time
import pandas as pd
from astroquery.hips2fits import hips2fits
from matplotlib.colors import Colormap

class ChiliAPT():
    """
    Class for simulate imaging stars on Chili focalplane

    Input Chili IFU pointing ra、dec and Position Angle, Return Star 
    Projection Location in IFU and Guiding Camera

    Parameters
    ----------
    ra_p : float
        ra pointing of the center of IFU/Guider, unit deg
        IFU/Guider中心的赤经指向，单位为度
        
    dec_p : float
        dec pointing of the center of IFU/Guider, unit deg
        IFU/Guider中心的赤纬指向，单位为度
        
    PA : float
        Position Angle of Focal plane, unit deg
        焦平面的位置角，单位为度
        
    center : str, optional
        Instrument set to center pointing, default is "IFU", optional "Guider"
        设置中心指向的仪器，默认为 "IFU", 可选 "Guider"
        
    observe_time : Time, optional
        Observation time, default is Time('2024-11-11')
        观测时间，默认为 Time('2024-11-11')
        
    save_path : str, optional
        Image save path, default is None
        图片保存路径，默认为 None
        格式为'./APT_result/',注意结尾要加'/'
    
    plot_sky : bool, optional
        Whether to plot sky map, default is True
        是否绘制天球图，默认为 True
        
    plot_stars : bool, optional
        Whether to plot star map, default is True
        是否绘制星图，默认为 True
        
    random_transform : bool, optional
        Whether to add random offset, default is False
        是否添加随机偏移，默认为 False
    
    Attributes
    ----------
    
    Methods
    -------

    """

    def __init__(self, ra_p, dec_p, PA, center = "IFU", observe_time = Time('2024-11-11'),
                  save_path=None, plot_sky = True, plot_stars = True, random_transform= False):
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        self.ra_p = ra_p
        self.dec_p = dec_p
        self.PA = PA
        self.phi_p = PA + 180
        self.observe_time = observe_time
        self.random_transform = random_transform
        self.save_path = save_path

        self.wcs = w = astropy_wcs.WCS(header={
            'NAXIS1':                 1200,  # Width of the output fits/image
            'NAXIS2':                 1200,  # Height of the output fits/image
            'WCSAXES':                   2,  # Number of coordinate axes
            'CRPIX1':                600.5, # Pixel coordinate of reference point
            'CRPIX2':                600.5,  # Pixel coordinate of reference point
            'CUNIT1': 'deg',                 # Units of coordinate increment and value
            'CUNIT2': 'deg',                 # Units of coordinate increment and value
            'CTYPE1': 'RA---TAN'          ,     # right ascension, gnomonic projection
            'CTYPE2': 'DEC--TAN'          ,     # declination, gnomonic projection
            'CRVAL1': self.ra_p                ,     # RA of reference point
            'CRVAL2': self.dec_p               ,     # DEC of reference point
            'CD1_1': -2/3600              ,     # coordinate transformation matrix element
            'CD1_2': 0                    ,     # coordinate transformation matrix element
            'CD2_1': 0,      # coordinate transformation matrix element
            'CD2_2': 2/3600      # coordinate transformation matrix element
        })

        self.hips_url = 'CDS/P/DSS2/color'
        import time
        from requests.exceptions import ReadTimeout

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.hipsimg = np.flipud(hips2fits.query_with_wcs(
                                        hips=self.hips_url,
                                        wcs=self.wcs,
                                        get_query_payload=False,
                                        format='jpg',
                                        min_cut=0,
                                        max_cut=99.9
                                        ))
                break
            except ReadTimeout:
                print(f"尝试 {attempt + 1} 失败，正在重试...")
                time.sleep(2)
        else:
            raise Exception("多次尝试后仍然无法获取hips图像")
        # 各个仪器的参数:
        '''
        导星视场 :  240.6 x 319.8 角秒 / 4.01 x 5.33角分 /  0.0668 x 0.0888 度
        Chili视场 :  65 x 71 角秒/ 1.083 * 1.183 角分 / 0.0180 x 0.0197 度
        两者距离   66.6mm / 11.93分 / 0.1989度
        底片比例:  10.74296875 角秒 /mm
        '''
        if center == "IFU":
            # IFU 
            self.xy0_IFU = (0,0)
            self.FOV_IFU = (0.0180, 0.0197)
            
            # Guider
            self.xy0_Guider = (0.1989, 0) 
            self.FOV_Guider = (0.0668, 0.0888)
        if center == "Guider":
            # Guider
            self.xy0_Guider = (0,0)
            self.FOV_Guider = (0.0668, 0.0888)
            # IFU
            self.xy0_IFU = (-0.1989, 0) 
            self.FOV_IFU = (0.0180, 0.0197)

        self.APT_main()

        if plot_sky is True:
            self.plot_sky()
            self.plot_IFUsky()
            self.plot_Guidersky()
        if plot_stars is True:
            self.plot_fp()
            self.plot_inst("IFU")
            self.plot_inst("Guider")
             

    """
    Part1: From sky to plane
    """
    # 1)天球坐标 → Chili本地天球坐标
    def sky_to_fpnativesky(self,ra,dec,ra_p,dec_p,phi_p):
        """
        ra:np.ndaary,row vector,unit deg.
        dec:np.ndarray,row vector,unit deg.
        """
        
        ra = np.deg2rad(ra)
        dec =  np.deg2rad(dec)
        ra_p = np.deg2rad(ra_p)
        dec_p = np.deg2rad(dec_p)
        phi_p =  np.deg2rad(phi_p)
        phi = np.rad2deg(phi_p+np.arctan2(-np.cos(dec)*np.sin(ra-ra_p),np.sin(dec)*np.cos(dec_p)
                                        -np.cos(dec)*np.sin(dec_p)*np.cos(ra-ra_p)))
        theta = np.rad2deg(np.arcsin(np.sin(dec)*np.sin(dec_p)+np.cos(dec)*np.cos(dec_p)
                        *np.cos(ra-ra_p)))
        return phi, theta
    
    # 2)焦面本地天球坐标 → 焦面投影平面坐标(理想坐标系)
    def fpnativesky_to_fpprojp(self,phi,theta):
        phi = np.deg2rad(phi)
        theta =  np.deg2rad(theta)
        def cos_sin(theta):
            cos_sin = np.zeros_like(theta)
            for i in range(len(theta)):
                if theta[i] == np.pi/2:
                    cos_sin[i] = 0
                else:
                    cos_sin[i] = np.cos(theta[i])/np.sin(theta[i])
            return cos_sin
        #R_theta = (180/np.pi)*(np.cos(theta)/np.sin(theta))
        R_theta = (180/np.pi)*cos_sin(theta)
        xi = R_theta*np.sin(phi)
        eta = - R_theta*np.cos(phi)
        return xi, eta
    
    def create_instrument_polygon(self, xi0_i, eta0_i, PA_i , fov , xi00_i_j = 0 , eta00_i_j = 0):
        """
        生成仪器的多边形
        
        """
        PA_i = np.deg2rad(PA_i)
        x_half_fov = fov[0] / 2
        y_half_fov = fov[1] / 2
        refpvec00 =  np.array([[xi00_i_j],[eta00_i_j]])
        
        vertex1_i = np.array([[- x_half_fov],[y_half_fov]]) + refpvec00
        vertex2_i = np.array([[x_half_fov],[y_half_fov]]) + refpvec00
        vertex3_i = np.array([[x_half_fov],[- y_half_fov]]) + refpvec00
        vertex4_i = np.array([[ - x_half_fov],[- y_half_fov]]) + refpvec00
        vec_vertex_i = np.column_stack([vertex1_i,vertex2_i,vertex3_i,vertex4_i])
        mat_trans_re = np.array([[np.cos(PA_i),np.sin(PA_i)],[-np.sin(PA_i),np.cos(PA_i)]])
        offset = np.array([[xi0_i],[eta0_i]])
        vec_vertice = mat_trans_re @ vec_vertex_i + offset
        inst_polygon = Polygon(vec_vertice.T)
        return inst_polygon
    
    def inside_fov(self,xi,eta,radec_table,inst_polygon):
        points = [Point(i, j) for i, j in zip(xi,eta)]
        bools = [inst_polygon.contains(Point(point)) for point in points]
        xi_in = xi[bools]
        eta_in = eta[bools]
        radec_table_in = radec_table[bools]
        return  xi_in, eta_in, radec_table_in
    
    def instrefpoint(self, xy0, random_transform = False):
        if random_transform is False:
            delta_xi0 = 0 
            delta_eta0 = 0
            PA_i = 0
        if random_transform is True:
            delta_xi0 = np.random.normal(loc=0, scale=2/3600) # sigma = 2 arcsec random x offset
            delta_eta0 = np.random.normal(loc=0, scale=2/3600) # sigma = 2 arcsec random y offset
            PA_i = np.random.normal(loc=0, scale= 1) # sigma = 500 arcsec random theta
        xi0 = xy0[0] + delta_xi0
        eta0 = - xy0[1] + delta_eta0
        return xi0 , eta0 , PA_i , delta_xi0*3600 , delta_eta0*3600

    # 焦面投影平面坐标 → 仪器投影平面坐标(平面变换)
    def fpprojp_to_instprojp(self,xi,eta,xi0_i,eta0_i,PA_i):
        PA_i = np.deg2rad(PA_i)
        vec_xi_eta = np.vstack([xi,eta])
        mat_trans = np.array([[np.cos(PA_i),-np.sin(PA_i)],[np.sin(PA_i),np.cos(PA_i)]])
        offset = np.array([[xi0_i],[eta0_i]])
        xi_i,eta_i = mat_trans@(vec_xi_eta-offset)
        return xi_i, eta_i

    def instprojp_to_instphy(self,xi_i, eta_i):
        xi_i = np.deg2rad(xi_i)
        eta_i = np.deg2rad(eta_i)
        f = 28000000  # unit:μm ,focal length
        x_i = xi_i*f
        y_i = eta_i*f
        return x_i , y_i
    
    """
    Part2: From plane to sky
    """
    def arg(self, x, y):
        if not isinstance(x, np.ndarray):
            x = np.array([x])
        if not isinstance(y, np.ndarray):
            y = np.array([y])
        arg=np.ones_like(x).astype(float)
        for i in range(len(x)):
            if (x[i]>=0)and(y[i]>=0):
                arg[i]=np.arctan2(y[i],x[i])
            elif (x[i]<0)and(y[i]>=0):
                arg[i]=np.arctan2(y[i],x[i])
            elif (x[i]<0)and(y[i]<0):
                arg[i]=np.arctan2(y[i],x[i])+2*np.pi
            elif (x[i]>=0)and(y[i]<0):
                arg[i]=np.arctan2(y[i],x[i])+2*np.pi
        return arg

    def fpprojp_to_fpnativesky(self, xi , eta):
        # Step 3
        phi_rad = self.arg(-eta , xi) # unit: rad 
        R = np.sqrt(xi**2 + eta**2) # unit :deg
        with np.errstate(divide='ignore'):
            theta_rad = np.arctan(180 / (np.pi * R))
        for i, r in enumerate(R):
            if r == 0:
                phi_rad[i] = 0
                theta_rad[i] = np.pi / 2
        phi =  np.rad2deg(phi_rad)
        theta = np.rad2deg(theta_rad)
        return phi, theta

    def fpnativesky_to_sky(self,phi, theta, ra_p, dec_p, phi_p):
        phi = np.deg2rad(phi)
        theta =  np.deg2rad(theta)
        phi_p = np.deg2rad(phi_p)
        dec_p = np.deg2rad(dec_p)
        t = np.arctan2(
            -np.cos(theta) * np.sin(phi - phi_p),
            np.sin(theta) * np.cos(dec_p) -
            np.cos(theta) * np.sin(dec_p) * np.cos(phi - phi_p))
        ra = ra_p + np.degrees(t)
        dec = np.degrees(
            np.arcsin(
                np.sin(theta) * np.sin(dec_p) +
                np.cos(theta) * np.cos(dec_p) * np.cos(phi - phi_p)))
        return ra, dec
    
    def polygon_to_sky(self, polygon_coords):
        """
        将多边形顶点坐标从焦面投影坐标转换到天球坐标系统
        
        Parameters
        ----------
        polygon_coords : list of tuples
            多边形顶点的焦面坐标 (xi, eta)
        
        Returns
        -------
        list of tuples
            多边形顶点的天球坐标 (ra, dec)
        """
        # 将坐标转换为numpy数组
        coords = np.array(polygon_coords)
        xi = coords[:, 0]
        eta = coords[:, 1]
        
        # 焦面投影坐标 -> 本地天球坐标
        phi, theta = self.fpprojp_to_fpnativesky(xi, eta)
        
        # 本地天球坐标 -> 天球坐标
        ra, dec = self.fpnativesky_to_sky(phi, theta, self.ra_p, self.dec_p, self.phi_p)
        
        return list(zip(ra, dec))

    def gaiadr3_query(self,ra: list,
                      dec: list,
                      rad: float = 1.0,
                      maxmag: float = 16,
                      maxsources: float = 10000000):
        """
        Acquire the Gaia DR3, from work of zhang tianmeng

        This function uses astroquery.vizier to query Gaia DR3 catalog.

        Parameters
        ----------
        ra : list
            RA of center in degrees.
        dec : list
            Dec of center in degrees.
        rad : float
            Field radius in degrees.
        maxmag : float
            Upper limit magnitude.
        maxsources : float
            Maximum number of sources.

        Returns
        -------
        astropy.table.Table
            Catalog of gaiadr3.

        Examples
        --------
        >>> catalog = gaiadr3_query([180.0], [30.0], 2.0)
        """

        vquery = Vizier(columns=[
            'RA_ICRS', 'DE_ICRS', 'pmRA', 'pmDE', 'Plx', 'RVDR2', 'Gmag',"BPmag","RPmag","BP-RP"
        ],
                        row_limit=maxsources,
                        column_filters={
                            "Gmag": ("<%f" % maxmag),
                            #"BPmag": ("<%f" % maxmag),
                            #"RPmag": ("<%f" % maxmag),
                            "Plx": ">0"
                        })
        coord = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame="icrs")
        r = vquery.query_region(coord,
                                radius=rad * u.deg,
                                catalog='I/355/gaiadr3')

        return r[0]

    def sky_coord(self,radectable):
        """

        Parameters
        ----------
        radec_table : astropy.table.Table
            Catalog of gaiadr3.

        Returns
        -------
        numpy 2d array
            gaia ra、dec
        """
        ra_s     = radectable["RA_ICRS"]
        dec_s    = radectable["DE_ICRS"]
        pmra_s   = radectable["pmRA"]
        pmdec_s  = radectable["pmDE"]
        paral_s  = radectable["Plx"]
        Gmag = radectable["Gmag"]
        if self.observe_time is None:
            ra = np.array(ra_s)
            dec = np.array(dec_s)
        else:
           c = SkyCoord(ra=ra_s, dec=dec_s,
                        distance=Distance(parallax=abs(paral_s) * u.mas),
                        pm_ra_cosdec=pmra_s,
                        pm_dec=pmdec_s,
                        obstime=Time(2016.0, format='jyear',
                                scale='tcb'), frame="icrs")
           epoch_observe = self.observe_time
           c_csst = c.apply_space_motion(epoch_observe)
           ra =  c_csst.ra.degree
           dec = c_csst.dec.degree
        return ra,dec, Gmag
    
    def APT_main(self):
        rad = 0.3 # query fov 0.3 degree
        radec_table = self.gaiadr3_query([self.ra_p],[self.dec_p], rad)
        ra,dec,self.Gmag = self.sky_coord(radec_table)

        self.radec_table = radec_table
        self.ra,self.dec = ra , dec 

        phi,theta = self.sky_to_fpnativesky(ra,dec,self.ra_p,self.dec_p,self.phi_p)
        self.xi, self.eta = self.fpnativesky_to_fpprojp(phi,theta)
        # Define reference point
        # Guider
        self.xi0_Guider,self.eta0_Guider,self.PA_Guider, self.delta_xi0_Guider, self.delta_eta0_Guider = self.instrefpoint(self.xy0_Guider, random_transform = self.random_transform)
        # IFU 
        self.xi0_IFU,self.eta0_IFU,self.PA_IFU, self.delta_xi0_IFU, self.delta_eta0_IFU = self.instrefpoint(self.xy0_IFU, random_transform = self.random_transform)
        
        # Create polygen
        self.IFU_polygon = self.create_instrument_polygon(self.xi0_IFU, self.eta0_IFU, self.PA_IFU ,self.FOV_IFU)
        self.Guider_polygon = self.create_instrument_polygon(self.xi0_Guider, self.eta0_Guider, self.PA_Guider ,self.FOV_Guider)
        self.IFU_polygon_sky = self.polygon_to_sky(list(self.IFU_polygon.exterior.coords[:-1]))
        self.Guider_polygon_sky = self.polygon_to_sky(list(self.Guider_polygon.exterior.coords[:-1]))
        # IFU 
        self.xi_in_IFU, self.eta_in_IFU, radec_table_in_IFU = self.inside_fov(self.xi, self.eta,radec_table,self.IFU_polygon)
        xi_IFU, eta_IFU = self.fpprojp_to_instprojp(self.xi_in_IFU,self.eta_in_IFU,self.xi0_IFU,self.eta0_IFU,self.PA_IFU)
        x_IFU, y_IFU = self.instprojp_to_instphy(xi_IFU, eta_IFU)
        self.Gmag_IFU = radec_table_in_IFU["Gmag"]
        self.phi0_IFU, self.theta0_IFU = self.fpprojp_to_fpnativesky(np.array([self.xi0_IFU]),np.array([self.eta0_IFU]))
        self.ra0_IFU,  self.dec0_IFU = self.fpnativesky_to_sky(self.phi0_IFU, self.theta0_IFU, self.ra_p, self.dec_p, self.phi_p)


        # Guider
        self.xi_in_Guider, self.eta_in_Guider, radec_table_in_Guider = self.inside_fov(self.xi, self.eta,radec_table,self.Guider_polygon)
        xi_Guider, eta_Guider = self.fpprojp_to_instprojp(self.xi_in_Guider,self.eta_in_Guider,self.xi0_Guider,self.eta0_Guider,self.PA_Guider)
        x_Guider, y_Guider =self.instprojp_to_instphy(xi_Guider, eta_Guider)
        self.Gmag_Guider = radec_table_in_Guider["Gmag"]
        self.phi0_Guider, self.theta0_Guider = self.fpprojp_to_fpnativesky(np.array([self.xi0_Guider]),np.array([self.eta0_Guider]))
        self.ra0_Guider,  self.dec0_Guider = self.fpnativesky_to_sky(self.phi0_Guider, self.theta0_Guider, self.ra_p, self.dec_p, self.phi_p)

        # IFU中心指向输出
        ifu_coord = SkyCoord(ra=self.ra0_IFU[0]*u.deg, dec=self.dec0_IFU[0]*u.deg, frame='icrs')
        self.ra_dec_IFU_text = f"IFU Center: RA={ifu_coord.ra.to_string(unit=u.hourangle, sep=':', precision=2)}, DEC={ifu_coord.dec.to_string(unit=u.deg, sep=':', precision=2)}"
        print(self.ra_dec_IFU_text)
        # Guider中心指向输出
        guider_coord = SkyCoord(ra=self.ra0_Guider[0]*u.deg, dec=self.dec0_Guider[0]*u.deg, frame='icrs')
        self.ra_dec_Guider_text = f"Guider Center: RA={guider_coord.ra.to_string(unit=u.hourangle, sep=':', precision=2)}, DEC={guider_coord.dec.to_string(unit=u.deg, sep=':', precision=2)}"
        print(self.ra_dec_Guider_text)

        pixel_size = 0.0668 / 700 # pixel size in degree of guider image
        self.guider_wcs = astropy_wcs.WCS(header={
            'NAXIS1':                 700 ,  # Width of the output fits/image
            'NAXIS2':                 934,  # Height of the output fits/image
            'WCSAXES':                   2,  # Number of coordinate axes
            'CRPIX1':                350.5, # Pixel coordinate of reference point
            'CRPIX2':                467.5,  # Pixel coordinate of reference point
            'CUNIT1': 'deg',                 # Units of coordinate increment and value
            'CUNIT2': 'deg',                 # Units of coordinate increment and value
            'CTYPE1': 'RA---TAN'          ,     # right ascension, gnomonic projection
            'CTYPE2': 'DEC--TAN'          ,     # declination, gnomonic projection
            'CRVAL1': self.ra0_Guider[0]                ,     # RA of reference point
            'CRVAL2': self.dec0_Guider[0]              ,     # DEC of reference point
            'CD1_1':  -pixel_size *np.cos(self.PA*np.pi/180)  ,# coordinate transformation matrix element
            'CD1_2':  pixel_size *np.sin(self.PA*np.pi/180)  ,     # coordinate transformation matrix element
            'CD2_1': pixel_size *np.sin(self.PA*np.pi/180),      # coordinate transformation matrix element
            'CD2_2': pixel_size *np.cos(self.PA*np.pi/180)   # coordinate transformation matrix element
        })
        
        ifu_pixel_size = 0.0180 / 650 # pixel size in degree of guider image
        self.ifu_wcs = astropy_wcs.WCS(header={
            'NAXIS1':                 650 ,  # Width of the output fits/image
            'NAXIS2':                 710 ,  # Height of the output fits/image
            'WCSAXES':                   2,  # Number of coordinate axes
            'CRPIX1':                325.5, # Pixel coordinate of reference point
            'CRPIX2':                355.5,  # Pixel coordinate of reference point
            'CUNIT1': 'deg',                 # Units of coordinate increment and value
            'CUNIT2': 'deg',                 # Units of coordinate increment and value
            'CTYPE1': 'RA---TAN'          ,     # right ascension, gnomonic projection
            'CTYPE2': 'DEC--TAN'          ,     # declination, gnomonic projection
            'CRVAL1': self.ra0_IFU[0]                ,     # RA of reference point
            'CRVAL2': self.dec0_IFU[0]              ,     # DEC of reference point
            'CD1_1':  -ifu_pixel_size *np.cos(self.PA*np.pi/180)  ,# coordinate transformation matrix element
            'CD1_2':  ifu_pixel_size *np.sin(self.PA*np.pi/180)  ,     # coordinate transformation matrix element
            'CD2_1': ifu_pixel_size *np.sin(self.PA*np.pi/180),      # coordinate transformation matrix element
            'CD2_2': ifu_pixel_size *np.cos(self.PA*np.pi/180)   # coordinate transformation matrix element
        })

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.guider_hips = np.flipud(hips2fits.query_with_wcs(
                    hips=self.hips_url,
                    wcs=self.guider_wcs,
                    get_query_payload=False,
                    format='jpg',
                    min_cut=0,
                    max_cut=99.9))
                break
            except ReadTimeout:
                print(f"获取Guider图像尝试 {attempt + 1} 失败，正在重试...")
                time.sleep(2)
        else:
            raise Exception("多次尝试后仍然无法获取Guider图像")

        for attempt in range(max_retries):
            try:
                self.ifu_hips = np.flipud(hips2fits.query_with_wcs(
                    hips=self.hips_url,
                    wcs=self.ifu_wcs,
                    get_query_payload=False,
                    format='jpg',
                    min_cut=0,
                    max_cut=99.9))
                break
            except ReadTimeout:
                print(f"获取IFU图像尝试 {attempt + 1} 失败，正在重试...")
                time.sleep(2)
        else:
            raise Exception("多次尝试后仍然无法获取IFU图像")

    def plot_fp(self):
        fig, ax = plt.subplots(figsize=(10,10))
        # 绘制相机视场
        plt.title("Chili Focal Plane", fontsize = 18)
        sizes = (max(self.Gmag) - self.Gmag + 1) * 10  # 根据Gmag调整点的大小，Gmag越小，size越大
        ax.scatter(self.xi, self.eta, c="black", marker=".", s=sizes, alpha=0.8)
        ax.add_patch(MatplotlibPolygon(list(self.IFU_polygon.exterior.coords[:-1]), facecolor='#FFB6C1', edgecolor='#8B0000',alpha =0.3))
        ax.add_patch(MatplotlibPolygon(list(self.Guider_polygon.exterior.coords[:-1]), facecolor='#ABEBC6', edgecolor='#2ECC40',alpha =0.3))
        
        # 设置文字
        ax.text(self.xi0_IFU, self.eta0_IFU-0.01,"IFU", fontsize=12, ha='center', va='center', color='black')
        ax.text(self.xi0_Guider, self.eta0_Guider,"Guider", fontsize=12, ha='center', va='center', color='black')
        # 设置坐标轴
        ax.set_xlabel("xi  (degree)", fontsize = 15)
        ax.set_ylabel("eta (degree)", fontsize = 15)
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-0.6, 0.6)
        ax.invert_xaxis()
        plt.axis('equal')
        plt.tight_layout()
        savename = self.save_path + "ChiliFocalPlane.jpg"
        plt.savefig(savename,dpi = 200)
        # 显示图形
        plt.show()
    
    def plot_sky(self):
        fig = plt.figure(figsize=(12, 12))  # 增加图像大小
        ax = plt.subplot(projection=self.wcs)
        ax.imshow(self.hipsimg, origin="lower")
        ax.add_patch(MatplotlibPolygon(self.IFU_polygon_sky*u.deg, facecolor='none', edgecolor='#8B0000', alpha=0.9, transform=ax.get_transform('icrs')))
        ax.text(self.IFU_polygon_sky[0][0], self.IFU_polygon_sky[0][1], 'IFU', color='#8B0000', transform=ax.get_transform('icrs'), fontsize=14)  # 增大文字大小
        ax.add_patch(MatplotlibPolygon(self.Guider_polygon_sky, facecolor='none', edgecolor='#2ECC40', alpha=0.9, transform=ax.get_transform('icrs')))
        ax.text(self.Guider_polygon_sky[0][0], self.Guider_polygon_sky[0][1], 'Guider', color='#2ECC40', transform=ax.get_transform('icrs'), fontsize=14)  # 增大文字大小
        # 将IFU和Guider的中心指向转换为小时角和度数格式并显示
        # IFU中心
        ax.text(0.95, 0.95, self.ra_dec_IFU_text, transform=ax.transAxes, fontsize=12, ha='right', va='top', color='white', bbox=dict(facecolor='black', alpha=0.5))
        # Guider中心
        ax.text(0.95, 0.90, self.ra_dec_Guider_text, transform=ax.transAxes, fontsize=12, ha='right', va='top', color='white', bbox=dict(facecolor='black', alpha=0.5))

        ax.grid(color='white', ls='solid', alpha=0.3)
        ax.set_xlabel('RA', fontsize=14)  # 增大标签文字大小
        ax.set_ylabel('DEC', fontsize=14)  # 增大标签文字大小
        plt.tight_layout()
        savename = self.save_path + "ChiliSky.jpg"
        plt.savefig(savename,dpi = 200)
        plt.show()

    def plot_IFUsky(self):
        fig = plt.figure(figsize=(12, 12))
        ax = plt.subplot(projection=self.ifu_wcs)
        ax.imshow(self.ifu_hips, origin="lower")
        ax.grid(color='white', ls='solid')
        ax.set_xlabel('RA', fontsize=14)
        ax.set_ylabel('DEC', fontsize=14)
        plt.title(self.ra_dec_IFU_text, fontsize=14, color='white', bbox=dict(facecolor='black', alpha=0.5))
        plt.tight_layout()
        if self.save_path is not None:
            savename = self.save_path + "IFUsky.jpg"
            plt.savefig(savename, dpi=200)
        plt.show()

    def plot_Guidersky(self):
        fig = plt.figure(figsize=(12, 12))
        ax = plt.subplot(projection=self.guider_wcs)
        ax.imshow(self.guider_hips, origin="lower")
        ax.grid(color='white', ls='solid')
        ax.set_xlabel('RA', fontsize=14)
        ax.set_ylabel('DEC', fontsize=14)
        plt.title(self.ra_dec_Guider_text, fontsize=14, color='white', bbox=dict(facecolor='black', alpha=0.5))
        plt.tight_layout()
        if self.save_path is not None:
            savename = self.save_path + "Guidersky.jpg"
            plt.savefig(savename, dpi=200)
        plt.show()
    
    def plot_inst_e(self,xi_in,eta_in,Gmag,inst_polygon,name):
        fig, ax = plt.subplots(figsize=(8,8))
        plt.title(name, fontsize = 18)
        ax.add_patch(MatplotlibPolygon(list(inst_polygon.exterior.coords[:-1]), facecolor='#FFB6C1', edgecolor='#8B0000',alpha =1))
        if len(Gmag) > 0:
            sizes = (max(Gmag) - Gmag + 1) * 10  # 根据Gmag调整点的大小，Gmag越小，size越大
            ax.scatter(xi_in, eta_in, c="k", marker="o", s=sizes, alpha=0.6)
        ax.set_xlabel("X  (degree)", fontsize = 15)
        ax.set_ylabel("Y (degree)", fontsize = 15)
        ax.invert_xaxis()
        plt.axis('equal')
        plt.tight_layout()
        if self.save_path is not None:
            savename = self.save_path + name + ".jpg"
            plt.savefig(savename,dpi = 200)
        # 显示图形
        plt.show()

    def plot_inst(self,name="IFU"):
        if name == "IFU":
            self.plot_inst_e(self.xi_in_IFU,self.eta_in_IFU,self.Gmag_IFU,self.IFU_polygon,name)
        if name == "Guider":
            self.plot_inst_e(self.xi_in_Guider,self.eta_in_Guider,self.Gmag_Guider,self.Guider_polygon,name)