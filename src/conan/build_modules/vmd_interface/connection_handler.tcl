#
# fconfigure $sock sets a few options for the generated socket,
# - "-blocking 0" keeps the program from hanging up if an empty
# command is entered
# - "-buffering line" reads lines of input (input stops at \n)
#
# fileevent $sock [list read_socket $sock] defines what happens when the socket reveives data
# - "readable" specifies the event (in this case, readable data is reveived)
# - "[list read_socket $sock]" list defines what should be done when the defined event happens
#
# the procedure (read_socket) checks wether any actual data was reveived (not empty).
# - "eval $data" then passes the reveiced command contained in $data and executes it in vmd
#
proc handle_connection {sock addr port} {
    fconfigure $sock -blocking 0 -buffering line
    fileevent $sock readable [list read_socket $sock]
}
proc read_socket {sock} {
    if {[eof $sock]} {
        close $sock
    } else {
        global mol_loaded
        set data [gets $sock]
        if {[string length $data] > 0} {
            if {[string match *load* $data]} {

                if {$mol_loaded == 0} {
                    # if no molecule is loaded we load one now and
                    # apply the standard representations
                    eval "mol delete all"
                    eval $data
                    set molid [molinfo list]
                    set molid [lindex $molid end]
                    mol delrep 0 $molid
                    mol representation CPK 1.0
                    mol color Element
                    mol addrep $molid
                    applyDisplaySettings
                    set structuralCount [countStructural structures/.tmp.xyz]
                    applyStructuralRepresentation $molid $structuralCount
                    set mol_loaded 1
                    } else {
                        # if there is already a structure loaded we do not want to simply reload it, as that
                        # would reset the camera angle. We can save the current vie with save_state.
                        save_state ".state.vmd"
                        eval "mol delete all"
                        # After deleting the old representation, we source the view state again
                        source ".state.vmd"
                        set molid [molinfo list]
                        set molid [lindex $molid end]
                        # delete the first two reps, since these were the ones added by
                        # applyStructuralRepresentation and are specific to the old structure.
                        mol delrep 0 $molid
                        mol delrep 1 $molid
                        # Now we apply the different representation to structural and functional
                        # atoms.
                        set structuralCount [countStructural structures/.tmp.xyz]
                        applyStructuralRepresentation $molid $structuralCount
                    }
                }
            }
            if {[string match *show_index* $data]} {
                # Create an atom selection for all atoms
                set sel [atomselect top "all"]

                # Get the indices and coordinates of all atoms in the selection
                set indices [$sel get index]
                set coords [$sel get {x y z}]

                # Specify the offset of the label. If we do not use an offset, the
                # numbers will be inside the atoms
                set z_offset 1

                # Loop through the atoms and draw a label for each, offset in the Z-direction
                for {set i 0} {$i < [llength $indices]} {incr i} {
                    # Extract the index and coordinates for the current atom
                    set index [lindex $indices $i]
                    set coord [lindex $coords $i]

                    # Modify the Z-coordinate by adding the offset
                    lset coord 2 [expr {[lindex $coord 2] + $z_offset}]

                    # Draw a label at the adjusted position showing its index
                    draw text $coord "$index"
            }
        }
    }
}
proc applyDisplaySettings {} {
    # Standard display settings, may be changed here
    color Display {Background} white
    display eyesep       0.065000
    display focallength  2.000000
    display height       6.000000
    display distance     -2.000000
    display projection   Perspective
    display nearclip set 0.500000
    display farclip  set 10.000000
    display depthcue   off
    display cuestart   0.500000
    display cueend     10.000000
    display cuestart   0.500000
    display cueend     10.000000
    display cuedensity 0.320000
    display cuemode    Exp2
    display shadows off
    display ambientocclusion off
    display aoambient 0.800000
    display aodirect 0.300000
    display dof off
    display dof_fnumber 64.000000
    display dof_focaldist 0.700000
}
proc applyStructuralRepresentation {molid structuralCount} {
    # This function applies different structural representations to
    # structural atoms and functionals groups.
    # First we apply a Bonds rep to all structural atoms
    mol rename $molid top
    mol delrep 0 top
    mol representation Bonds 0.300000 12.000000
    mol color Name
    mol selection "index < $structuralCount"
    mol material Opaque
    mol addrep top
    mol selupdate 0 top 0
    mol colupdate 0 top 0
    mol scaleminmax top 0 0.000000 0.000000
    mol smoothrep top 0 0
    mol drawframes top 0 {now}
    mol clipplane center 0 0 top {0.0 0.0 0.0}
    mol clipplane color  0 0 top {0.5 0.5 0.5 }
    mol clipplane normal 0 0 top {0.0 0.0 1.0}
    mol clipplane status 0 0 top {0}
    mol clipplane center 1 0 top {0.0 0.0 0.0}
    mol clipplane color  1 0 top {0.5 0.5 0.5 }
    mol clipplane normal 1 0 top {0.0 0.0 1.0}
    mol clipplane status 1 0 top {0}
    mol clipplane center 2 0 top {0.0 0.0 0.0}
    mol clipplane color  2 0 top {0.5 0.5 0.5 }
    mol clipplane normal 2 0 top {0.0 0.0 1.0}
    mol clipplane status 2 0 top {0}
    mol clipplane center 3 0 top {0.0 0.0 0.0}
    mol clipplane color  3 0 top {0.5 0.5 0.5 }
    mol clipplane normal 3 0 top {0.0 0.0 1.0}
    mol clipplane status 3 0 top {0}
    mol clipplane center 4 0 top {0.0 0.0 0.0}
    mol clipplane color  4 0 top {0.5 0.5 0.5 }
    mol clipplane normal 4 0 top {0.0 0.0 1.0}
    mol clipplane status 4 0 top {0}
    mol clipplane center 5 0 top {0.0 0.0 0.0}
    mol clipplane color  5 0 top {0.5 0.5 0.5 }
    mol clipplane normal 5 0 top {0.0 0.0 1.0}
    mol clipplane status 5 0 top {0}
    mol representation CPK 1.000000 0.300000 12.000000 12.000000
    mol color Name
    # Next we apply a CPK rep to all other atoms
    mol selection "index >= $structuralCount"
    mol material Opaque
    mol addrep top
    mol selupdate 1 top 0
    mol colupdate 1 top 0
    mol scaleminmax top 1 0.000000 0.000000
    mol smoothrep top 1 0
    mol drawframes top 1 {now}
    mol clipplane center 0 1 top {0.0 0.0 0.0}
    mol clipplane color  0 1 top {0.5 0.5 0.5 }
    mol clipplane normal 0 1 top {0.0 0.0 1.0}
    mol clipplane status 0 1 top {0}
    mol clipplane center 1 1 top {0.0 0.0 0.0}
    mol clipplane color  1 1 top {0.5 0.5 0.5 }
    mol clipplane normal 1 1 top {0.0 0.0 1.0}
    mol clipplane status 1 1 top {0}
    mol clipplane center 2 1 top {0.0 0.0 0.0}
    mol clipplane color  2 1 top {0.5 0.5 0.5 }
    mol clipplane normal 2 1 top {0.0 0.0 1.0}
    mol clipplane status 2 1 top {0}
    mol clipplane center 3 1 top {0.0 0.0 0.0}
    mol clipplane color  3 1 top {0.5 0.5 0.5 }
    mol clipplane normal 3 1 top {0.0 0.0 1.0}
    mol clipplane status 3 1 top {0}
    mol clipplane center 4 1 top {0.0 0.0 0.0}
    mol clipplane color  4 1 top {0.5 0.5 0.5 }
    mol clipplane normal 4 1 top {0.0 0.0 1.0}
    mol clipplane status 4 1 top {0}
    mol clipplane center 5 1 top {0.0 0.0 0.0}
    mol clipplane color  5 1 top {0.5 0.5 0.5 }
    mol clipplane normal 5 1 top {0.0 0.0 1.0}
    mol clipplane status 5 1 top {0}
    mol representation Bonds 0.100000 12.000000
    mol color Name
    # Last, we need to define a Bond rep with thin bonds for all atoms.
    # This needs to be done since functional and structural atoms would otherwise
    # not be connected
    mol selection {all}
    mol material Opaque
    mol addrep top
    mol selupdate 2 top 0
    mol colupdate 2 top 0
    mol scaleminmax top 2 0.000000 0.000000
    mol smoothrep top 2 0
    mol drawframes top 2 {now}
    mol clipplane center 0 2 top {0.0 0.0 0.0}
    mol clipplane color  0 2 top {0.5 0.5 0.5 }
    mol clipplane normal 0 2 top {0.0 0.0 1.0}
    mol clipplane status 0 2 top {0}
    mol clipplane center 1 2 top {0.0 0.0 0.0}
    mol clipplane color  1 2 top {0.5 0.5 0.5 }
    mol clipplane normal 1 2 top {0.0 0.0 1.0}
    mol clipplane status 1 2 top {0}
    mol clipplane center 2 2 top {0.0 0.0 0.0}
    mol clipplane color  2 2 top {0.5 0.5 0.5 }
    mol clipplane normal 2 2 top {0.0 0.0 1.0}
    mol clipplane status 2 2 top {0}
    mol clipplane center 3 2 top {0.0 0.0 0.0}
    mol clipplane color  3 2 top {0.5 0.5 0.5 }
    mol clipplane normal 3 2 top {0.0 0.0 1.0}
    mol clipplane status 3 2 top {0}
    mol clipplane center 4 2 top {0.0 0.0 0.0}
    mol clipplane color  4 2 top {0.5 0.5 0.5 }
    mol clipplane normal 4 2 top {0.0 0.0 1.0}
    mol clipplane status 4 2 top {0}
    mol clipplane center 5 2 top {0.0 0.0 0.0}
    mol clipplane color  5 2 top {0.5 0.5 0.5 }
    mol clipplane normal 5 2 top {0.0 0.0 1.0}
    mol clipplane status 5 2 top {0}
    mol rename top $molid
}
proc countStructural {filePath} {
    # Open the file for reading
    set file [open $filePath r]
    set count 0

    # Read the file line by line
    while {[gets $file line] != -1} {
        # Split the line into a list of words
        set words [split $line]

        # Check if the fifth column (index 4) is "structural"
        if {[lindex $words 4] eq "Structure"} {
            incr count
        }
    }

    # Close the file
    close $file

    # Return the count
    return $count
}
