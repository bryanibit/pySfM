# pySfM
Simplify OpenSfM open source, suitable for beginner of SfM
The process is based on the procedure of OpenSfM (python)--extract EXIF(Output is a text document), 
detect feature(Use pickle.dump to save feature including (x,y)/color(r,g,b) and feature ID), match feature(Use pickle.load to load 
detected feature and pickle.dump to save matches), create track, 
