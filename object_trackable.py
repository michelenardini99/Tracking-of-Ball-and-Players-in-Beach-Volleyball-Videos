import numpy as np

class TrackableObject:
   def __init__(self, objectID, centroid):
      # store the object ID, then initialize a list of centroids
      # using the current centroid
      self.objectID = objectID
      self.centroids = [centroid]
      self.color = list(np.random.choice(range(256), size=3))
      # initialize a boolean used to indicate if the object has
      # already been counted or not
      self.counted = False