#!/usr/bin/env python3
"""
Canevas.py — Entry point for the ACO-TSP interactive GUI.

Launch with:
    python Canevas.py
"""

from aco_tsp.gui import TSPApp

if __name__ == "__main__":
    app = TSPApp()
    app.run()
    def not_used_keep_deplacement(self):
        self.__can.move(self.__canid, 0, 0)
        
class Line:
    def __init__(self, canvas, dx, dy, ax, ay, color):
        self.__dx, self.__dy, self.__ax, self.__ay=dx,dy,ax, ay
        self.__can= canvas
        self.__color= color
        self.__canid = self.__can.create_line(dx,dy,ax,ay, fill=color)
    def get_line_ident(self):
        return self.__canid
    
    


# ------------------------------------------------------
# Réutilisation des formes

# --------------------------------------------------------
if __name__ == "__main__":
    fen = FenPrincipale()
    fen.mainloop()
