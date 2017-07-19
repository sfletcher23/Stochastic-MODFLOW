import flopy
import os
from osgeo import goal

if os.path.exits("dem.tiff"):
    modName = 'mod'
    mf1 = flopy.modflow.Modflow(modName, exe_name=   )
