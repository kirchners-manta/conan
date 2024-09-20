#!/usr/bin/python3.10

import tkinter as tk


def on_button_click():
    print("Button was clicked!")


def on_button_click2():
    print("Button2 was clicked!")
    # close the window
    root.destroy()


# Hauptfenster erstellen
root = tk.Tk()
root.title("Event Example")

# Button erstellen
button = tk.Button(root, text="Click me", command=on_button_click)

# zweiten Button erstellen
button2 = tk.Button(root, text="Close me", command=on_button_click2)

# Button platzieren
button.pack()
button2.pack()

# Event-Schleife starten
root.mainloop()
