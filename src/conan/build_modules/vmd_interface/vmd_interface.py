import os
import socket
import subprocess

import conan.defdict as ddict


def start_vmd():
    """
    This function starts VMD in a new terminal
    """
    # First we get the path to the tcl scripts
    tcl_path = os.path.abspath(os.path.dirname(__file__)) + "/"

    # VMD is started in a new terminal, this avoids problems with the conan and vmd
    # prompts blocking each other
    process_id = subprocess.Popen(
        ["gnome-terminal", "--", "vmd", "-e", tcl_path + "socket_main.tcl", "-args", tcl_path]
    )

    # We return the process id so we can try to forcefully shut VMD down later if
    # we need to
    return process_id


def update_structure():
    """
    This function sends the command to load the structure saved in structures/.tmp.xyz
    to the VMD server.
    """
    send_command("mol load xyz structures/.tmp.xyz")  # load structure from tmp file


def send_command(vmd_command):
    """
    This function opens a connection to the specified socket and sends the
    command given in vmd_command as plain text to it. At this point, VMD has to
    be started with the tcl_server listening to the specified port.
    """
    host = "localhost"
    # The port value is hardcoded in socke_main.tcl. If the value here is changed
    # it has to be changed there aswell!
    port = 48654
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            sock.sendall(vmd_command.encode("utf-8"))
    except ConnectionError as e:
        ddict.printLog(f"Failed to connect to VMD: {e}")
