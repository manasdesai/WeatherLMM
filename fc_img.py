import pandas as pd
import numpy as np
import argparse
import os
import gc
import psutil
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def make_fc_img(date, init_time, lead_time, levels):
    """
    Generate all ECMWF-style forecast images for a single date.

    Parameters
    ----------
    date : str           # "YYYYMMDD"
    init_time : str      # "0000" or "1200"
    lead_time : int      # hours (integer)
    """
    import metview as mv
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
   
    # =====================================================================
    # File locations
    # =====================================================================
    year = date[0:4]
    month = date[4:6]
    day = date[6:8]

    hour = (int(init_time[:2]) + int(lead_time)) % 24
    hour_str = f"{hour:02}00"

    t2m_file = f"/glade/campaign/collections/rda/data/d113001/ec.oper.an.sfc/{year}{month}/ec.oper.an.sfc.128_167_2t.regn1280sc.{year}{month}{day}.grb"
    u10_file = f"/glade/campaign/collections/rda/data/d113001/ec.oper.an.sfc/{year}{month}/ec.oper.an.sfc.128_165_10u.regn1280sc.{year}{month}{day}.grb"
    v10_file = f"/glade/campaign/collections/rda/data/d113001/ec.oper.an.sfc/{year}{month}/ec.oper.an.sfc.128_166_10v.regn1280sc.{year}{month}{day}.grb"
    msl_file = f"/glade/campaign/collections/rda/data/d113001/ec.oper.an.sfc/{year}{month}/ec.oper.an.sfc.128_151_msl.regn1280sc.{year}{month}{day}.grb"

    z_file = f"/glade/campaign/collections/rda/data/d113001/ec.oper.an.pl/{year}{month}/ec.oper.an.pl.128_129_z.regn1280sc.{year}{month}{day}{hour:02}.grb"
    t_file = f"/glade/campaign/collections/rda/data/d113001/ec.oper.an.pl/{year}{month}/ec.oper.an.pl.128_130_t.regn1280sc.{year}{month}{day}{hour:02}.grb"
    u_file = f"/glade/campaign/collections/rda/data/d113001/ec.oper.an.pl/{year}{month}/ec.oper.an.pl.128_131_u.regn1280uv.{year}{month}{day}{hour:02}.grb"
    v_file = f"/glade/campaign/collections/rda/data/d113001/ec.oper.an.pl/{year}{month}/ec.oper.an.pl.128_132_v.regn1280uv.{year}{month}{day}{hour:02}.grb"
    r_file = f"/glade/campaign/collections/rda/data/d113001/ec.oper.an.pl/{year}{month}/ec.oper.an.pl.128_157_r.regn1280sc.{year}{month}{day}{hour:02}.grb"

    savedir = f"/glade/derecho/scratch/dcalhoun/ecmwf/ifs/charts/{init_time}/{lead_time}/{year}/{month}/{day}"
    os.makedirs(savedir, exist_ok=True)

    style_dict = {
        "t2m_wind10m": (),
        "thickness_mslp":  ()
    }

    for level in levels:
        style_dict.update({f"t_z_{level}": ()})

    if all([os.path.exists(os.path.join(savedir, f"{key}_{init_time}_{lead_time}_{date}.1.png")) for key in style_dict.keys()]):
        logging.info(f"All images exist for {year} {month} {day} {hour:02}UTC, skipping.")
        return

    # =====================================================================
    # Read GRIB
    # =====================================================================
    logging.info(f"Reading GRIB fields for {year} {month} {day} {hour:02}UTC")
    t2m = mv.read(t2m_file).select(time=hour_str)
    u10 = mv.read(u10_file).select(time=hour_str)
    v10 = mv.read(v10_file).select(time=hour_str)
    msl = mv.read(msl_file).select(time=hour_str)

    z = mv.read(z_file)
    t = mv.read(t_file)
    u = mv.read(u_file)
    v = mv.read(v_file)
    r = mv.read(r_file)
    
    # =====================================================================
    # Define styles
    # =====================================================================
    coast = mv.mcoast(
        map_coastline_colour="charcoal",
        map_coastline_resolution="medium",
        map_coastline_land_shade="on",
        map_coastline_land_shade_colour="cream",
        map_coastline_sea_shade="off",
        map_boundaries="on",
        map_boundaries_colour="charcoal",
        map_grid_colour="tan",
        map_label_height=0.35,
    )

    view = mv.geoview(
        area_mode="name",
        area_name="north_america",
        subpage_clipping="on",
        subpage_x_length=78,
        subpage_y_length=68,
        subpage_x_position=17,
        subpage_y_position=12,
        coastlines=coast,
    )

    ecmwf_text = mv.mtext(
        text_lines=[
            "© European Centre for Medium-Range Weather Forecasts (ECMWF)",
            "Source: www.ecmwf.int Licence: CC-BY-4.0 and ECMWF Terms of Use",
            "https://apps.ecmwf.int/datasets/licences/general/",
        ],
        text_justification="center",
        text_font_size=0.4,
        text_mode="positional",
        text_box_x_position=11,
        text_box_y_position=0.5,
        text_box_x_length=8,
        text_box_y_length=2,
        text_colour="charcoal",
    )

    legend = mv.mlegend(legend_text_font_size=0.4)

    t2m_wind10m_title = mv.mtext(
        text_lines = ["2m temperature and 10m wind",
                    "VALID TIME: <grib_info key='valid-date' format='%a %d %B %Y %H' where='shortName=2t' />",
                     ""],
        text_font_size = 0.4,
        text_colour = 'charcoal')
    t2m_shade = mv.mcont(legend= "on",
                    contour_automatics_settings = "style_name",
                    contour_style_name = "sh_all_fM48t56i4")
    
    msl_thickness_title = mv.mtext(
        text_lines=["500-1000 hPa thickness and Mean sea level pressure", 
               "VALID TIME: <grib_info key='valid-date' format='%a %d %B %Y %H' where='shortName=msl' />",
                   ""],
        text_font_size=0.4,
        text_colour         = 'charcoal')
    thickness_shade = mv.mcont(legend= "on",
                    contour_automatics_settings = "style_name",
                    contour_style_name = "sh_blured_fM16t50_thickness")
    msl_shade = mv.mcont(legend= "on",
                    contour_automatics_settings = "style_name",
                    contour_style_name = "ct_blk_i5_t2")
    
    t_z_title = mv.mtext(
        text_lines=["Temperature and geopotential at various pressure levels, level <grib_info key='level' where='shortName=t' /> hPa ", 
                    "VALID TIME: <grib_info key='valid-date' format='%a %d %B %Y %H' where='shortName=t'/>",
                   ""],
        text_font_size=0.4,
        text_colour         = 'charcoal')
    t_shade = mv.mcont(legend= "on",
                    contour_automatics_settings = "style_name",
                    contour_style_name = "sh_all_fM64t52i4")
    
    z_shade = mv.mcont(legend= "off",
                    contour_automatics_settings = "style_name",
                    contour_style_name = "ct_blk_i5_t2")

    uv_rh_title = mv.mtext(
        text_lines=["Wind and relative humidity at various pressure levels, level <grib_info key='level' where='shortName=r' /> hPa ", 
               "VALID TIME: <grib_info key='valid-date' format='%a %d %B %Y %H' where='shortName=r'/>",
                   ""],
        text_font_size=0.4,
        text_colour         = 'charcoal')
    rh_shade = mv.mcont(legend= "on",
                    contour_automatics_settings = "style_name",
                    contour_style_name = "sh_grnblu_f65t100i15")

    def wind_arrows(scale):
        wind_arrows = mv.mwind(
            wind_thinning_factor=scale, wind_arrow_colour="black"
        )
        return wind_arrows

    # =====================================================================
    # Build a style dictionary and make plots
    # =====================================================================

    def lev(field, lev):
        return field.select(level=lev)
    
    t2m -= 273.15
    wind10m = mv.grib_vectors(u_component=u10, v_component=v10)
    style_dict.update({"t2m_wind10m" : (t2m_wind10m_title, t2m, t2m_shade, wind10m, wind_arrows(10))})
    
    thickness = (lev(z, 500) - lev(z, 1000)) / 98.06
    msl /= 100
    style_dict.update({"thickness_mslp" : (msl_thickness_title, msl, msl_shade, thickness, thickness_shade)})

    for level in levels:
        temp = lev(t, level) - 273.15
        hgt = lev(z, level) / 98.06
        style_dict.update({f"t_z_{level}": (t_z_title, temp, t_shade, hgt, z_shade)})

        rh = lev(r, level)
        uu = lev(u, level)
        vv = lev(v, level)
        wind =  mv.grib_vectors(u_component=uu, v_component=vv)
        scale = np.floor(((6/np.log10(level))**3.35))
        style_dict.update({f"uv_rh_{level}": (uv_rh_title, rh, rh_shade, wind, wind_arrows(scale))})

    def plot(key):
        outfile = os.path.join(savedir, f"{key}_{init_time}_{lead_time}_{date}")
        fields = style_dict[key]
        png = mv.png_output(
            output_name=outfile,
            output_title=os.path.basename(outfile),
            output_width=1000,
        )
        mv.setoutput(png)
        mv.plot(view, *fields, ecmwf_text, legend)
        logging.info(f"Saved {outfile}.1.png")

    for k in style_dict.keys():
        plot(k)

    # Explicit cleanup of Metview objects to release GRIB handles
    for obj in [
        t2m, u10, v10, wind10m, msl, thickness, z, t, u, v, wind, r 
    ]:
        try:
            del obj
        except Exception:
            pass
        
    gc.collect()
    
    proc = psutil.Process()
    logging.info(f"Finished {year}-{month}-{day} {hour:02}UTC | RSS={proc.memory_info().rss/1e9:.2f} GB")
    return

def run_task(date, init_time, lead_time):
    cmd = ["python", "-u", "worker.py", "--date", date, "--init_time", init_time, "--lead_time", str(lead_time)]
    proc = subprocess.run(cmd, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr
    
def run_concurrent(start, end, init_times, lead_time, levels=[1000, 850, 700, 500, 200], num_workers=24):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    dates = pd.date_range(start=start, end=end, freq='D')
    with ThreadPoolExecutor(max_workers=24) as exc:
        for date in dates:
            for init_time in init_times:
                futures = [exc.submit(run_task, date.strftime("%Y%m%d"), init_time, lead_time)]
        for f in as_completed(futures):
            ret, out, err = f.result()

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=True, help="start date YYYYMMDD")
    parser.add_argument('--end', required=True, help="end date YYYYMMDD")
    parser.add_argument('--init_times', required=True, help="HHHH[, ...]")
    parser.add_argument('--lead_time', required=True, help="lead hours as string or int")
    args = parser.parse_args()
    init_times = args.init_times.split(",")
    run_concurrent(args.start, args.end, init_times, args.lead_time)

if __name__ == "__main__":
    main()