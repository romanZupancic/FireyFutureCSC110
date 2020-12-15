"""
This file uses the subprocess library to initialize the website

Copyright and Usage Information
===============================
This file is Copyright (c) 2020 Daniel Hocevar and Roman Zupancic. 

This files contents may not be modified or redistributed without written
permission from Daniel Hocevar and Roman Zupancic
"""

import subprocess

def main() -> None:
    """
    Call < streamlit run main.py > to start the streamlit website
    """
    subprocess.run(['streamlit', 'run', 'streamlit_main.py'])


if __name__ == '__main__':
    main()