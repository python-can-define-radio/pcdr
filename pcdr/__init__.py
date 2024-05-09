try:
    from pcdr import flow
   
except ModuleNotFoundError:
    print("WARNING: Unable to import gnuradio-related functionality.")
    print("You can still use other functions, such as the modulators.")
    print("If you wish to install GNU Radio, see this page: https://wiki.gnuradio.org/index.php/InstallingGR")
