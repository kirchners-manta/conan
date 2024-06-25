# Get path of directories with all needed tcl files.
# Since this script is started with the VMD tcl interpreter
# neither "info script", nor $::argv0 work here.
# the path is simply passed directly by the python program.
set tcl_dir [lindex $argv 0]
set mol_loaded 0

source [file join $tcl_dir "connection_handler.tcl"]

# The port value is also hardcoded in vmd_interface.py. If the value here is changed
# it also has to be changed in that file aswell!
set port 48654
# - set sock generates a socket and saves its ID in the variable "sock"
# - "-server handle_connections" starts the procedure "handle_connections"
# as soon as something connects to the socket
set sock [socket -server handle_connection $port]
puts "VMD socket started on port $port..."
