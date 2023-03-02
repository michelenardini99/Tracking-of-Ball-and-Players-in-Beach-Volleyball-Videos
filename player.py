

class Player:

    def __init__(self,name,coord):
        self.name=name
        self.coord=coord

    def setPos(self,coord):
        self.coord=coord

    def getCoord(self):
        return self.coord

    def getName(self):
        return self.name

    def __str__(self):
        return f"{self.name}({self.coord})"