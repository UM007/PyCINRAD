# -*- coding: utf-8 -*-
# Author: gym

import datetime

import numpy as np
from xarray import Dataset, DataArray

from cinrad.projection import get_coordinate, height
from cinrad.constants import deg2rad
from cinrad.io.base import RadarBase, prepare_file
from cinrad.io.level3 import StandardPUP, ProductParamsParser
from cinrad.io._dtype import *
from cinrad.error import RadarDecodeError

__all__ = ["Standard_X_PUP"]


def arr_to_dictlist(arr):
    fileds = arr.dtype.names
    data_dict = [dict(zip(fileds, a)) for a in arr]

    return data_dict


class Standard_X_PUP(StandardPUP):
    # fmt: off
    dtype_corr = {1: 'TR', 2: 'REF', 3: 'V', 4: 'SW', 5: 'SQI', 6: 'CPA', 7: 'ZDR', 8: 'LDR',
                  9: 'CC', 10: 'PDP', 11: 'KDP', 12: 'CP', 14: 'HCL', 15: 'CF', 16: 'SNRH',
                  17: 'SNRV', 32: 'Zc', 33: 'Vc', 34: 'Wc', 35: 'ZDRc', 36: 'PDP', 37: 'KDP',
                  38: 'RHO', 71: 'RR', 72: 'HGT', 73: 'VIL', 74: 'SHR', 75: 'RAIN', 76: 'RMS',
                  77: 'CTR'}

    ptype_corr = {1: "PPI", 2: "RHI", 3: "CAR", 4: "MAX", 6: "ET", 8: "VCS",
                  9: "LRA", 10: "LRM", 13: "SRR", 14: "SRM", 20: "WER", 23: "VIL",
                  24: "HSR", 25: "OHP", 26: "THP", 27: "STP", 28: "NHP", 31: "VAD",
                  32: "VWP", 34: "Shear", 36: "SWP", 37: "STI", 38: "HI", 39: "M",
                  40: "TVS", 41: "SS", 48: "GAGE", 51: "HCL", 52: "QPE", 18: "CR", 53: "EB", 54: "CR",
                  44: "UAM", 50: "ML", 55: "WIND", 56: "RCH", 58: "HMAX"}

    # fmt: on
    def __init__(self, file):
        self.f = prepare_file(file)
        self._parse_header()
        self._update_radar_info()
        self.stationlat = self.geo["lat"][0].item()
        self.stationlon = self.geo["lon"][0].item()
        self.radarheight = self.geo["height"][0].item()
        if self.name == "None":
            self.name = self.code
        del self.geo
        if self.ptype in [1, 13, 14, 24, 25, 26, 27, 28, 51, 52]:  # PPI radial format
            self._parse_radial_fmt()
        elif self.ptype in [4, 6, 8, 9, 10, 18, 23, 53, 54, 56, 58]:
            self._parse_raster_fmt()
        elif self.ptype == 3:
            self._parse_car_fmt()
        elif self.ptype == 20:
            self._parse_wer_fmt()
        elif self.ptype == 31:
            self._parse_vad_fmt()
        elif self.ptype == 32:
            self._parse_vwp_fmt()
        elif self.ptype == 36:
            self._parse_swp_fmt()
        elif self.ptype == 37:
            self._parse_sti_fmt()
        elif self.ptype == 38:
            self._parse_hail_fmt()
        elif self.ptype == 39:
            self._parse_m_fmt()
        elif self.ptype == 40:
            self._parse_tvs_fmt()
        elif self.ptype == 41:
            self._parse_ss_fmt()
        elif self.ptype == 44:
            self._parse_uam_fmt()
        elif self.ptype == 50:
            self._parse_ml_fmt()
        elif self.ptype == 55:
            self._parse_wind_fmt()
        # elif self.ptype == 56:
        #     self._parse_rch_fmt()
        else:
            raise RadarDecodeError(
                "Unsupported product type {}:{}".format(self.ptype, self.pname)
            )
        self.f.close()

    def _parse_header(self):
        header = np.frombuffer(self.f.read(32), SDD_header)
        if header["magic_number"] != 0x4D545352:
            raise RadarDecodeError("Invalid standard data")
        site_config = np.frombuffer(self.f.read(128), SDD_site)
        # for ss in site_config.dtype.names:
        #     print(ss, site_config[ss])
        self.code = site_config["site_code"][0].decode().replace("\x00", "")
        self.geo = geo = dict()
        geo["lat"] = site_config["Latitude"]
        geo["lon"] = site_config["Longitude"]
        geo["height"] = site_config["ground_height"]
        self.radar_type = site_config["radar_type"][0]
        task = np.frombuffer(self.f.read(256), SDD_task)
        self.task_name = task["task_name"][0].decode().replace("\x00", "")
        cut_num = task["cut_number"][0]
        self.scan_config = np.frombuffer(self.f.read(256 * cut_num), SDD_cut)
        ph = np.frombuffer(self.f.read(128), SDD_pheader)
        self.ptype = ph["product_type"][0]
        self.pname = self.ptype_corr[self.ptype]
        proj_type = ph["proj_type"][0]  # 投影类型 1:麦卡托投影,2:等距方位投影,13:兰勃特方位等积投影
        self.scantime = datetime.datetime.fromtimestamp(
            ph["scan_start_time"][0], datetime.timezone.utc
        )
        if self.ptype == 1:
            self.pname = self.dtype_corr[ph["dtype_1"][0]]
        self.params = ProductParamsParser.parse(self.ptype, self.f.read(64))

    def _parse_radial_fmt(self):
        radial_header = np.frombuffer(self.f.read(64), L3_radial)
        bin_length = radial_header["bin_length"][0]
        scale = radial_header["scale"][0]
        offset = radial_header["offset"][0]
        reso = radial_header["reso"][0] / 1000
        start_range = radial_header["start_range"][0] / 1000
        end_range = radial_header["max_range"][0] / 1000
        nradial = radial_header["nradial"][0]
        data = list()
        azi = list()
        for _ in range(nradial):
            buf = self.f.read(32)
            if not buf:
                break
            data_block = np.frombuffer(buf, L3_rblock)
            start_a = data_block["start_az"][0]
            nbins = data_block["nbins"][0]
            raw = np.frombuffer(
                self.f.read(bin_length * nbins), "u{}".format(bin_length)
            )
            data.append(raw)
            azi.append(start_a)
        if nradial == 0:
            da = []
        else:
            raw = np.vstack(data).astype(int)
            data_rf = np.ma.masked_not_equal(raw, 1)
            raw = np.ma.masked_less(raw, 5)
            data = (raw - offset) / scale
            if self.ptype in [25, 26, 27, 28]:
                # Mask 0 value in precipitation products
                data = np.ma.masked_equal(data, 0)
            az = np.linspace(0, 360, raw.shape[0])
            az += azi[0]
            az[az > 360] -= 360
            azi = az * deg2rad
            # self.azi = np.deg2rad(azi)
            if self.radar_type == 12:
                dist = np.linspace(start_range + reso, start_range + reso * raw.shape[1], raw.shape[1])
            else:
                dist = np.arange(start_range + reso, end_range + reso * 1.1, reso)
            lon, lat = get_coordinate(
                dist, azi, self.params["elevation"], self.stationlon, self.stationlat
            )
            hgt = (
                    height(dist, self.params["elevation"], self.radarheight)
                    * np.ones(azi.shape[0])[:, np.newaxis]
            )
            da = DataArray(data, coords=[azi, dist], dims=["azimuth", "distance"])

        ds = Dataset(
            {self.pname: da},
            attrs={
                "elevation": self.params["elevation"],
                "range": end_range,
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "tangential_reso": reso,
                "task": self.task_name,
            },
        )
        if nradial > 0:
            ds["longitude"] = (["azimuth", "distance"], lon)
            ds["latitude"] = (["azimuth", "distance"], lat)
            ds["height"] = (["azimuth", "distance"], hgt)
            if self.pname in ["V", "VEL", "SW"]:
                ds["RF"] = (["azimuth", "distance"], data_rf)
        self._dataset = ds

    def _parse_raster_fmt(self):
        raster_header = np.frombuffer(self.f.read(64), L3_raster)
        bin_length = raster_header["bin_length"][0].item()
        scale = raster_header["scale"][0]
        offset = raster_header["offset"][0]
        reso = raster_header["row_reso"][0] / 1000
        nx = raster_header["row_side_length"][0].item()
        ny = raster_header["col_side_length"][0].item()
        raw = (
            np.frombuffer(self.f.read(nx * ny * bin_length), "u{}".format(bin_length))
            .reshape(nx, ny)
            .astype(int)
        )
        raw = np.ma.masked_less(raw, 5)
        data = (raw - offset) / scale
        max_range = int(nx / 2 * reso)
        y = np.linspace(max_range, max_range * -1, ny) / 111 + self.stationlat
        x = (
                np.linspace(max_range * -1, max_range, nx) / (111 * np.cos(y * deg2rad))
                + self.stationlon
        )
        lon, lat = np.meshgrid(x, y)
        da = DataArray(
            data,
            coords=[lat[:, 0], lon[0]],
            dims=["latitude", "longitude"],
        )
        ds = Dataset(
            {self.pname: da},
            attrs={
                "elevation": 0,
                "range": max_range,
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "tangential_reso": reso,
            },
        )
        self._dataset = ds

    def _parse_car_fmt(self):
        azi = list()
        dist = list()
        data = list()
        height = np.linspace(
            self.params["bottom"], self.params["top"], self.params["layers_count"]
        )
        for _ in range(self.params["layers_count"]):
            radial_header = np.frombuffer(self.f.read(64), L3_radial)
            bin_length = radial_header["bin_length"][0]
            scale = radial_header["scale"][0]
            offset = radial_header["offset"][0]
            reso = radial_header["reso"][0] / 1000
            start_range = radial_header["start_range"][0] / 1000
            end_range = radial_header["max_range"][0] / 1000 + 1
            # for dd in radial_header.dtype.names:
            #     print(dd, radial_header[dd][0])
            nradial = radial_header["nradial"][0]
            azi0 = list()
            for __ in range(nradial):
                buf = self.f.read(32)
                if not buf:
                    break
                data_block = np.frombuffer(buf, L3_rblock)
                start_a = data_block["start_az"][0]
                nbins = data_block["nbins"][0]
                raw = np.frombuffer(
                    self.f.read(bin_length * nbins), "u{}".format(bin_length)
                )
                data.append(raw)
                azi0.append(start_a)
            if len(azi) == 0:
                raw = np.vstack(data).astype(int)
                az = np.linspace(0, 360, raw.shape[0])
                az += azi0[0]
                az[az > 360] -= 360
                azi = az * deg2rad
                dist = np.arange(start_range + reso, end_range + reso * 0.1, reso)
                # print(len(dist))
        raw = np.vstack(data).astype(int)
        raw = np.ma.masked_less(raw, 5)
        dist = dist[:raw.shape[-1]]
        data = (raw - offset) / scale
        data = np.reshape(data, (self.params["layers_count"], len(azi), -1))
        lon, lat = get_coordinate(dist, azi, 0, self.stationlon, self.stationlat)
        da = DataArray(
            data, coords=[height, azi, dist], dims=["height", "azimuth", "distance"]
        )
        ds = Dataset(
            {self.pname: da},
            attrs={
                "elevation": 0,
                "range": end_range,
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "tangential_reso": reso,
                "task": self.task_name,
            },
        )
        ds["longitude"] = (["azimuth", "distance"], lon)
        ds["latitude"] = (["azimuth", "distance"], lat)

        self._dataset = ds

    def _parse_vwp_fmt(self):
        self.vwp_header = np.frombuffer(self.f.read(32), L3_vwp_header)
        timestamp = list()
        height = list()
        wd = list()
        ws = list()
        rms = list()
        while True:
            buf = self.f.read(32)
            if not buf:
                break
            vwp = np.frombuffer(buf, L3_vwp)
            timestamp.append(vwp["start_time"][0])
            height.append(vwp["height"][0])
            wd.append(vwp["wind_direction"][0])
            ws.append(vwp["wind_speed"][0])
            rms.append(vwp["rms_std"][0])
        height = list(set(height))
        timestamp = list(set(timestamp))
        height.sort()
        timestamp.sort()
        shape = (len(timestamp), len(height))
        wd = np.round(np.array(wd).astype(float).reshape(shape), 0)
        ws = np.round(np.array(ws).astype(float).reshape(shape), 2)
        rms = np.round(np.array(rms).astype(float).reshape(shape), 2)
        wd_da = DataArray(
            wd,
            coords=[
                timestamp,
                height,
            ],
            dims=["times", "height"],
        )
        ds = Dataset(
            {"wind_direction": wd_da},
            attrs={
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "task": self.task_name,
            },
        )
        ds["wind_speed"] = (["times", "height"], ws)
        ds["rms"] = (["times", "height"], rms)
        self._dataset = ds

    def _parse_swp_fmt(self):
        swp_count = np.frombuffer(self.f.read(4), "i4")[0]
        swp = np.frombuffer(self.f.read(swp_count * 12), L3_swp)
        swp_azimuth = np.array(swp["azimuth"])
        swp_range = np.array(swp["range"])[:, np.newaxis]
        swp_percent = DataArray(swp["swp"])
        lon, lat = get_coordinate(
            swp_range / 1000,
            swp_azimuth * deg2rad,
            self.params["elevation"],
            self.stationlon,
            self.stationlat,
        )
        ds = Dataset(
            {
                "swp_percent": swp_percent,
            },
            attrs={
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "task": self.task_name,
            },
        )
        ds["longitude"] = DataArray(lon[:, 0])
        ds["latitude"] = DataArray(lat[:, 0])
        self._dataset = ds

    def _parse_hail_fmt(self):
        hail_count = np.frombuffer(self.f.read(4), "i4")[0]
        hail_table = np.frombuffer(self.f.read(hail_count * 28), L3_hail)
        ht0msl = np.frombuffer(self.f.read(4), "f4")[0]
        ht20msl = np.frombuffer(self.f.read(4), "f4")[0]
        hail_azimuth = np.array(hail_table["hail_azimuth"])
        hail_range = np.array(hail_table["hail_range"])[:, np.newaxis]
        hail_size = DataArray(hail_table["hail_size"])
        hail_possibility = DataArray(hail_table["hail_possibility"])
        hail_severe_possibility = DataArray(hail_table["hail_severe_possibility"])
        lon, lat = get_coordinate(
            hail_range / 1000,
            hail_azimuth * deg2rad,
            self.params["elevation"],
            self.stationlon,
            self.stationlat,
        )
        ds = Dataset(
            {
                "hail_possibility": hail_possibility,
                "hail_size": hail_size,
                "hail_severe_possibility": hail_severe_possibility,
            },
            attrs={
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "task": self.task_name,
                "height_0deg": ht0msl,
                "height_-20deg": ht20msl,
            },
        )
        ds["longitude"] = DataArray(lon[:, 0])
        ds["latitude"] = DataArray(lat[:, 0])
        self._dataset = ds

    def _parse_m_fmt(self):
        storm_count = np.frombuffer(self.f.read(4), "i4")[0]
        m_count = np.frombuffer(self.f.read(4), "i4")[0]
        feature_count = np.frombuffer(self.f.read(4), "i4")[0]

        m_table = np.frombuffer(self.f.read(m_count * 68), L4_m)
        m_list = arr_to_dictlist(m_table)

        feature_table = np.frombuffer(self.f.read(feature_count * 72), L4_feature)
        feature_list = arr_to_dictlist(feature_table)

        dataAdapter = np.frombuffer(self.f.read(52), L4_m_dataAdapter)
        dataAdapter = arr_to_dictlist(dataAdapter)[0]



        # fmt: on
        ds = {
            "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
            "site_code": self.code,
            "site_name": self.name,
            "site_longitude": self.stationlon,
            "site_latitude": self.stationlat,
            "task": self.task_name,
            'storm_count': storm_count,
            'm_count': m_count,
            'm_list': m_list,
            'feature_count': feature_count,
            'feature_list': feature_list,
            'dataAdapter': dataAdapter
        }

        self._dataset = ds

    def _parse_tvs_fmt(self):
        tvs_count = np.frombuffer(self.f.read(4), "i4")[0]
        etvs_count = np.frombuffer(self.f.read(4), "i4")[0]
        tvs_table = np.frombuffer(self.f.read((tvs_count + etvs_count) * 56), L3_tvs)
        minrefl = np.frombuffer(self.f.read(4), "i4")[0]
        minpvdv = np.frombuffer(self.f.read(4), "i4")[0]
        tvs_azimuth = np.array(tvs_table["tvs_azimuth"])
        tvs_range = np.array(tvs_table["tvs_range"])[:, np.newaxis]
        lon, lat = get_coordinate(
            tvs_range / 1000,
            tvs_azimuth * deg2rad,
            self.params["elevation"],
            self.stationlon,
            self.stationlat,
        )

        data_dict = {}
        # fmt: off
        for key in ["tvs_id", "tvs_stormtype", "tvs_azimuth", "tvs_range", "tvs_elevation",
                    "tvs_lldv", "tvs_avgdv", "tvs_mxdv", "tvs_mxdvhgt", "tvs_depth", "tvs_base",
                    "tvs_top", "tvs_mxshr", "tvs_mxshrhgt"]:
            data_dict[key] = DataArray(tvs_table[key])
        # fmt: on
        attrs_dict = {
            "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
            "site_code": self.code,
            "site_name": self.name,
            "site_longitude": self.stationlon,
            "site_latitude": self.stationlat,
            "task": self.task_name,
            "minrefl": minrefl,
            "minpvdv": minpvdv,
        }
        ds = Dataset(data_dict, attrs=attrs_dict)
        ds["longitude"] = DataArray(lon[:, 0])
        ds["latitude"] = DataArray(lat[:, 0])
        self._dataset = ds

    def _parse_sti_fmt(self):
        sti_header = np.frombuffer(self.f.read(20), L3_sti_header)
        sti_count = sti_header["num_of_storms"][0].item()
        track_count = sti_count if sti_count < 100 else 100
        sti_current = np.frombuffer(self.f.read(24 * track_count), L3_sti_motion)
        curr_azimuth = np.array(sti_current["azimuth"])
        curr_range = np.array(sti_current["range"])[:, np.newaxis]
        curr_speed = sti_current["speed"]
        curr_direction = sti_current["direction"]
        curr_lon, curr_lat = get_coordinate(
            curr_range / 1000,
            curr_azimuth * deg2rad,
            self.params["elevation"],
            self.stationlon,
            self.stationlat,
        )
        curr = [
            [curr_lon[i, 0], curr_lat[i, 0], curr_speed[i], curr_direction[i]]
            for i in range(track_count)
        ]
        forecast = []
        for _ in range(track_count):
            forecast_positon_count = np.frombuffer(self.f.read(4), "i4")[0]
            forecast_positon = np.frombuffer(
                self.f.read(12 * forecast_positon_count), L3_sti_position
            )
            fore_azimuth = np.array(forecast_positon["azimuth"])
            fore_range = np.array(forecast_positon["range"])[:, np.newaxis]
            fore_lon, fore_lat = get_coordinate(
                fore_range / 1000,
                fore_azimuth * deg2rad,
                self.params["elevation"],
                self.stationlon,
                self.stationlat,
            )
            fore = [
                [fore_lon[i, 0], fore_lat[i, 0]] for i in range(forecast_positon_count)
            ]
            forecast.append(fore)
        history = []
        for _ in range(track_count):
            history_positon_count = np.frombuffer(self.f.read(4), "i4")[0]
            history_positon = np.frombuffer(
                self.f.read(12 * history_positon_count), L3_sti_position
            )
            his_azimuth = np.array(history_positon["azimuth"])
            his_range = np.array(history_positon["range"])[:, np.newaxis]
            his_lon, his_lat = get_coordinate(
                his_range / 1000,
                his_azimuth * deg2rad,
                self.params["elevation"],
                self.stationlon,
                self.stationlat,
            )
            his = [[his_lon[i, 0], his_lat[i, 0]] for i in range(history_positon_count)]
            history.append(his)
        self.sti_attributes = []
        self.sti_components = []
        if sti_count > 0:
            for _ in range(track_count):
                self.sti_attributes.append(
                    np.frombuffer(self.f.read(60), L3_sti_attribute)
                )
            for _ in range(track_count):
                self.sti_components.append(
                    np.frombuffer(self.f.read(12), L3_sti_component)
                )
            self.sti_adaptation = np.frombuffer(self.f.read(40), L3_sti_adaptation)
        sti_id = [attr["id"] for attr in self.sti_attributes]
        max_ref = [attr["max_ref"] for attr in self.sti_attributes]
        max_ref_height = [attr["max_ref_height"] for attr in self.sti_attributes]
        vil = [attr["vil"] for attr in self.sti_attributes]
        top_height = [attr["top_height"] for attr in self.sti_attributes]
        sti_data = [
            {
                "id": str(sti_id[i][0]),
                "current_position": [curr[i][0], curr[i][1]],
                "current_speed": curr[i][2],
                "current_direction": curr[i][3],
                "forecast_position": forecast[i],
                "history_position": history[i],
                "max_ref": max_ref[i][0],
                "max_ref_height": max_ref_height[i][0],
                "vil": vil[i][0],
                "top_height": top_height[i][0],
            }
            for i in range(track_count)
        ]
        attrs = {
            "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
            "site_code": self.code,
            "site_name": self.name,
            "site_longitude": self.stationlon,
            "site_latitude": self.stationlat,
            "task": self.task_name,
            "sti_count": sti_count,
        }
        ds = {"data": sti_data, "attrs": attrs}
        self._dataset = ds

    def _parse_uam_fmt(self):
        uam_count = np.frombuffer(self.f.read(4), "i4")[0]
        uam_table = np.frombuffer(self.f.read(uam_count * 44), L3_uam)
        uam_azimuth = np.array(uam_table["azimuth"])
        uam_range = np.array(uam_table["range"])[:, np.newaxis]
        uam_a = DataArray(uam_table["a"])
        uam_b = DataArray(uam_table["b"])
        uam_deg = DataArray(uam_table["deg"])
        lon, lat = get_coordinate(
            uam_range / 1000,
            uam_azimuth * deg2rad,
            self.params["elevation"],
            self.stationlon,
            self.stationlat,
        )
        ds = Dataset(
            {
                "a": uam_a,
                "b": uam_b,
                "deg": uam_deg,
            },
            attrs={
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "task": self.task_name,
            },
        )
        ds["longitude"] = DataArray(lon[:, 0])
        ds["latitude"] = DataArray(lat[:, 0])
        self._dataset = ds

    def _parse_wer_fmt(self):
        wer = Dataset()
        while True:
            buf = self.f.read(32)
            if not buf:
                break
            wer_header = np.frombuffer(buf, L3_wer_header)
            elev = wer_header["elevation"][0]
            self._parse_raster_fmt()
            if len(wer) == 0:
                wer = self._dataset.copy()
                wer = wer.rename({self.pname: "{}_{:.1f}".format(self.pname, elev)})
                wer.attrs["center_height"] = wer_header["center_height"][0]
            else:
                wer["{}_{:.1f}".format(self.pname, elev)] = self._dataset[self.pname]
        self._dataset = wer

    def _parse_vad_fmt(self):
        vat_header = np.frombuffer(self.f.read(64), L3_vad_header)
        attrs = arr_to_dictlist(vat_header)[0]
        attrs.pop('res')

        number_data_points = vat_header["number_data_points"][0]

        vad_table = np.frombuffer(self.f.read(number_data_points * 12), L3_vad)

        vad_azimuth = np.array(vad_table["azimuth"])
        vad_velocity = np.array(vad_table["velocity"])
        vad_reflectivity = np.array(vad_table["reflectivity"])
        vad = Dataset(
            {
                "a": vad_azimuth,
                "v": vad_velocity,
                "ref": vad_reflectivity,
            },
            attrs=attrs,
        )
        self._dataset = vad

    # def _parse_wind_fmt(self):
    #     wind_header = np.frombuffer(self.f.read(56), L3_wind_header)
    #     attrs = {}
    #     for w in wind_header.dtype.names[:-1]:
    #         attrs[w] = wind_header[w][0]
    #
    #     code_len = int(attrs['Row_Side_Length']) * int(attrs['Column_Side_Length']) * 4
    #     wind_table = np.frombuffer(self.f.read(code_len), L3_wind)
    #     raw_speed = np.array(wind_table["speed"])
    #     raw_speed = np.ma.masked_less(raw_speed, attrs['Speed_Offset'])
    #
    #     raw_direction = np.array(wind_table["direction"])
    #     raw_direction = np.ma.masked_less(raw_direction, attrs['Direction_Offset'])
    #
    #     nx = attrs['Row_Side_Length'] // attrs['Row_Resolution']
    #     ny = attrs['Column_Side_Length'] // attrs['Column_Resolution']
    #     reso = attrs['Row_Resolution']
    #
    #     wind_speed_data = (raw_speed - attrs['Speed_Offset']) / attrs['Speed_Scale']
    #     wind_speed_data = np.reshape(wind_speed_data, (nx, ny))
    #
    #     wind_direction_data = (raw_direction - attrs['Direction_Offset']) / attrs['Direction_Scale']
    #     wind_direction_data = np.reshape(wind_direction_data, (nx, ny))
    #
    #     max_range = int(nx / 2000 * reso)
    #     y = np.linspace(max_range, max_range * -1, ny) / 111 + self.stationlat
    #     x = np.linspace(max_range * -1, max_range, nx) / (111 * np.cos(y * deg2rad)) + self.stationlon
    #     lon, lat = np.meshgrid(x, y)
    #     speed = DataArray(
    #         wind_speed_data,
    #         coords=[lat[:, 0], lon[0]],
    #         dims=["latitude", "longitude"],
    #     )
    #     direction = DataArray(
    #         wind_direction_data,
    #         coords=[lat[:, 0], lon[0]],
    #         dims=["latitude", "longitude"],
    #     )
    #     ds = Dataset(
    #         {"wind_speed": speed, "wind_direction": direction},
    #         attrs={
    #             "elevation": 0,
    #             "range": max_range,
    #             "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
    #             "site_code": self.code,
    #             "site_name": self.name,
    #             "site_longitude": self.stationlon,
    #             "site_latitude": self.stationlat,
    #             "tangential_reso": reso,
    #         },
    #     )
    #     self._dataset = ds
    def _parse_wind_fmt(self):
        wind_header = np.frombuffer(self.f.read(56), L3_wind_header)
        attrs = {}
        for w in wind_header.dtype.names[:-1]:
            attrs[w] = wind_header[w][0]

        code_len = int(attrs['Row_Side_Length']) * int(attrs['Column_Side_Length']) * 4
        wind_table = np.frombuffer(self.f.read(code_len), L3_wind)
        raw_speed = np.array(wind_table["speed"])
        raw_speed = np.ma.masked_less(raw_speed, attrs['Speed_Offset'])

        raw_direction = np.array(wind_table["direction"])
        raw_direction = np.ma.masked_less(raw_direction, attrs['Direction_Offset'])

        nx = attrs['Row_Side_Length'] // attrs['Row_Resolution']
        ny = attrs['Column_Side_Length'] // attrs['Column_Resolution']
        reso = attrs['Row_Resolution']

        wind_speed_data = (raw_speed - attrs['Speed_Offset']) / attrs['Speed_Scale']
        # 将masked数据置为0
        # wind_speed_data = np.ma.filled(wind_speed_data, 0)
        # wind_speed_data = np.reshape(wind_speed_data, (nx, ny))

        wind_direction_data = (raw_direction - attrs['Direction_Offset']) / attrs['Direction_Scale']
        # wind_direction_data = np.reshape(wind_direction_data, (nx, ny))

        max_range = int(nx / 2000 * reso)
        lat = np.linspace(max_range, max_range * -1, ny) / 111 + self.stationlat
        lon = np.linspace(max_range * -1, max_range, nx) / (111 * np.cos(lat * deg2rad)) + self.stationlon
        # lon, lat = np.meshgrid(x, y)

        ds = {
                "elevation": 0,
                "range": max_range,
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "tangential_reso": reso,
                'left': lon.min(), 'right': lon.max(), 'top': lat.max(), 'bottom': lat.min(), 'rows': ny, 'cols': nx,
                'height': 0, 'noDataValue': None,
                "speed": wind_speed_data,
                "direction": wind_direction_data
            }
        self._dataset = ds

    def _parse_ml_fmt(self):
        try:
            # 融化层点阵数量
            ml_count = np.frombuffer(self.f.read(4), 'i4')[0].item()
            ml = arr_to_dictlist(np.frombuffer(self.f.read(ml_count * L3_ml.itemsize), ml_count * L3_ml)[0])

        except IndexError:
            ml = []

        self._dataset = ml

    def _parse_ss_fmt(self):
        # nstorms = int.from_bytes(self.f.read(4), byteorder='big')
        ss_count = np.frombuffer(self.f.read(4), 'i4')[0].item()

        ss_table = []
        ss_cell_trend = []
        if ss_count:
            tables = np.frombuffer(self.f.read(ss_count * 32), L3_ss_table)
            for st in tables:
                stdict = {}
                for k in st.dtype.names:
                    stdict[k] = st[k]
                ss_table.append(stdict)

            for i in range(ss_count):
                storm_id = np.frombuffer(self.f.read(4), 'i4')[0].item()
                nvolumes = np.frombuffer(self.f.read(4), 'i4')[0].item()
                trend_dtype = np.dtype([
                    ("volume_time", "i4"),
                    ("height", "i4"),
                    ("base_height", "i4"),
                    ("top_height", "i4"),
                    ("vil", "i4"),
                    ("maximum_reflectivity", "i4"),
                    ("height_of_maximum_reflectivity", "i4"),
                    ("possibility_of_hail", "i4"),
                    ("possibility_of_severe_hail", "i4"),
                ])

                trend_tables = arr_to_dictlist(np.frombuffer(self.f.read(nvolumes * 36), trend_dtype))
                trend_info = {'storm_id': storm_id,
                              'number_of_volumes': nvolumes,
                              'trend_tables': trend_tables, }

                ss_cell_trend.append(trend_info)

        segment_data = arr_to_dictlist(np.frombuffer(self.f.read(96), L3_ss_cell_segment))[0]

        ss_centroids_adaptation = arr_to_dictlist(np.frombuffer(self.f.read(92), L3_ss_centroids))[0]

        ss_track_adaptation = arr_to_dictlist(np.frombuffer(self.f.read(40), L3_sti_adaptation))[0]

        ss_data = {'ss_table': ss_table,
                   'ss_trend': ss_cell_trend,
                   'segment_data': segment_data,
                   'ss_centroids_adaptation': ss_centroids_adaptation,
                   'ss_track_adaptation': ss_track_adaptation}
        attrs = {
            "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
            "site_code": self.code,
            "site_name": self.name,
            "site_longitude": self.stationlon,
            "site_latitude": self.stationlat,
            "task": self.task_name,
            "sti_count": ss_count,
        }
        ds = {'ss_data': ss_data, "attrs": attrs}
        self._dataset = ds

    def get_data(self):
        return self._dataset
