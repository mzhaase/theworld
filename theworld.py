import time
import os
import logging
import io
import math

from datetime import datetime

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# from pyproj import Transformer
from mpl_toolkits.basemap import Basemap

logging.basicConfig(level=logging.DEBUG)

# if not using raspberry and not having GPIO
if os.uname()[4] != 'armv6l':
  TESTING = True
else:
  TESTING =  False

NEOPIXEL = True

BRIGHTNESS = 0.1
# Dimensions of the pixel matrix
ROWS       = 17
COLUMNS    = 33

# Our Map:
LOWER_LAT = -60
UPPER_LAT = 75
LEFT_LON  = -180
RIGHT_LON = 180

# For conversion from meters lat / lon to pixels
LAT_METERS = 20048966.104014594
LON_METERS = 20037508.342789244
# we use a coordinate system centered on 0 meridian / equator, therefore
# a quarter of the pixels are in every sector and we need to divide them
# by 2
LAT_METERS_PER_PIXEL = LAT_METERS / (ROWS / 2)
LON_METERS_PER_PIXEL = LON_METERS / (COLUMNS / 2)
# TRAN_4326_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857")

# sample coords:
COORD_BERLIN = (13.404954, 52.520008)
COORD_DC = (-77.0369,38.9072)
COORD_OSLO = (10.7522, 59.9139)
COORD_ANCHORAGE = (-149.9003, 61.2181)

if not TESTING:
  if NEOPIXEL:
    import board
    import neopixel
    Pixels  = neopixel.NeoPixel(
      board.D18,
      ROWS * COLUMNS,
      auto_write  = False,
      pixel_order = neopixel.GRB,
      brightness  = BRIGHTNESS
    )
  else:
    from ws2812 import WS2812
    WS2812 = WS2812(
      intensity = BRIGHTNESS
    )



logging.info(f'Testing is {TESTING}')

class OutOfRangeError(Exception):
  """Exception raised values that are out of range."""
  pass

def clear():
  """Clears all pixels
  """
  if not TESTING:
    if NEOPIXEL:
      Pixels.fill((0,0,0))
      Pixels.show()
    else:
      ws_data = []
      for i in range(ROWS * COLUMNS):
        ws_data.append((0,0,0))
      WS2812.show(ws_data)
  return

def validate_geo_coord(lon, lat):
  """Make sure coordinates are valid

  Args:
      lon (float): Longitude
      lat (float): Latitude

  Raises:
      OutOfRangeError: If coordinate not within map
  """
  if lon > 360 or lon < -360:
    raise OutOfRangeError()
  # web mercator only goes to 85.06 degrees N/S
  if lat > 85.06 or lat < -85.06:
    raise OutOfRangeError()

def euclidean_coord_to_pixel(x: int, y: int, centered=True):
  """Takes euclidiean coords and returns the corresponding
  pixel id

  Args:
      x (int): x coordinate
      y (int): y coordinate

  Returns:
      int: Pixel ID
  """
  if centered:
    origin = int(round((COLUMNS*ROWS) / 2))
    pixel_id  = origin - (y * COLUMNS) + x
  else:
    pixel_id = y * COLUMNS + x
  return pixel_id


def geo_coord_to_euclidean(lon, lat):
  """Takes geo coordinates as degrees and converts them to
  x, y coordinates on our pixel matrix

  Args:
      lon (float): Longitude, E positive 0-180, W negative 0 - -180
      lat (float): Latitude, N positive 0-85.06, S negative 0- -85.06

  Returns:
      int,int: x, y coordinates
  """  
  validate_geo_coord(lon, lat)
  x_meters, y_meters = TRAN_4326_TO_3857.transform(lat, lon)
  x = int(round(x_meters / LON_METERS_PER_PIXEL, 0))
  y = int(round(y_meters / LAT_METERS_PER_PIXEL, 0))
  return x, y

def geo_coord_to_pixel(lon, lat):
  """For a given longitude, latitude, return the pixel

  Args:
      lon (float): Longitude (West/East) 0 - 360
      lat (float): Latitude (North/South) -90 - 90

  Returns:
      [int]: Pixel ID that corresponds to these coordinats
  """
  validate_geo_coord(lon, lat)

  x, y = geo_coord_to_euclidean(lon, lat)
  return euclidean_coord_to_pixel(x, y)

def calculate_terminator(date):
  # todo: calculate equinox
  equinox = datetime(year=2021,month=3,day=20,hour=9,minute=37)
  day_offset = 

def main():
  clear()
  map = Basemap(
    projection = 'merc',
    llcrnrlat  = LOWER_LAT,
    llcrnrlon  = LEFT_LON,
    urcrnrlat  = UPPER_LAT,
    urcrnrlon  = RIGHT_LON
  )

  # map.drawcoastlines()
  # map.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
  # map.drawmeridians(np.arange(map.lonmin,map.lonmax+30,60),labels=[0,0,0,1])

  # map.drawmapboundary(fill_color='aqua')
  # map.fillcontinents(color='coral',lake_color='aqua')


  while True:
    start_time = time.time()
    ws_data = []
    fig = plt.figure(
      figsize   = [COLUMNS, ROWS],
      dpi       = 1,
      frameon   = False
    )
    # shamelessly stolen from
    # https://stackoverflow.com/a/53310715
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
      hspace = 0, wspace = 0)
    plt.margins(0,0)
    logging.info('beginning new cycle')
    # start_time += 7200
    # date = datetime.fromtimestamp(start_time)
    # time.sleep(1)
    date = datetime.utcnow()
    CS   = map.nightshade(date, alpha=1)
    logging.debug(f'Terminator calculation time: {time.time() - start_time}')
    logging.debug('draw figure')
    if TESTING:
      fig.savefig('test.png', tranparent=True)
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw', tranparent=True)
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    for y, row in enumerate(im):
      #logging.debug(f'Row: {y} {row}')
      for x, color in enumerate(row):
        if x == 0 or x == COLUMNS - 1: continue
        color = tuple(color[:3])
        if color[0] > 0:
          color = (156,156,255)
        if NEOPIXEL:
          pixel_id = euclidean_coord_to_pixel(x, y, centered=False)
          if not TESTING:
            Pixels[pixel_id] = color
        else:
          ws_data.append(color)
    if not TESTING:
      if NEOPIXEL:
        Pixels.show()
      else:
        WS2812.show(ws_data)
    plt.close()
    logging.debug(f'Cycle time: {time.time() - start_time}')
    start_time = time.time()
    # print(geo_coord_to_pixel(COORD_ANCHORAGE[0], COORD_ANCHORAGE[1]))

if __name__ == '__main__':
  main()
