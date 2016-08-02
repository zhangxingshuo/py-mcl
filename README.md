# Monte Carlo Localization
Monte Carlo Localization implemented in Python.

Provides support for ORB, SIFT, and SURF image matching algorithms, with addition of optimization techniques such as DOR (Dynamically Optimized Retrieval) and BOW (Bag-of-Words)

Also includes visualization tools and performance benchmarks.

## Setup
The images from the robot camera should be stored in a folder named `cam1_img` inside the root directory. Also inside the root directory should be a text file `commands.txt` that contains the image index and corresponding command of the robot -- l, r, f, b, s (left, right, forward, backward, and stop, respectively). 

## Image Matching
The script `Matcher.py` provides support for a multitude of image matching algorithms. For example, to match a query image `query.jpg` against the first location in the map -- the folder `map/0` -- using SIFT (Scale Invariant Feature Transform) at 320x240 resolution, run

`python -i Matcher.py`

in CLI. Next, instantiate a Matcher object with the desired arguments.

`>> matcher = Matcher('SIFT',320,240)`

Set the query image and dataset directory.
```
>> matcher.setQuery('query.jpg')
>> matcher.setDirectory('map/0')
```
Finally, run the image matching. 

`>> matcher.run()`

This returns the total matches for that image, and the list of normalized probabilities that each correspond to an image in the map.

## Matching an image sequence
The script `analyze.py` provides a framework for matching large sequences of images and running the Monte Carlo Localization algorithm. In CLI, run

`python -i analyze.py`

and instantiate an analyzer object.

`>> analyzer = analzyer('SIFT',320,240)`

To create the output file of raw probability lists, run

`>> analyzer.createRawP()`

This writes the probability lists for all images in the sequence to the file `rawP.txt`. Note that at high resolutions, this could take a long time. On an 2.5 GHz Intel i5 processor, a sequence of 200 images took approximately 2 hours to run using SIFT at 800x600.

To run the Monte Carlo Localization algorithm, simply run

`>> analyzer.processRaw()`

Note that this does not do any matching; rather, it reads from the `rawP.txt` created in the step before. Thus, one could store different output files to save time and processing power. This method creates a file called `out.txt`, which has the adjusted probability lists.

The algorithm first measures the sharpness of the image using the variance of the Laplacian, a method described [here](http://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/). It assigns a weight to the current update proportional to the variance of the Laplacian. Next, it adjusts the particles according to the command that is read from `commands.txt` using Python list splicing. Lastly, the previous generation is factored into the current update. 

To visualize the algorithm, run

`python GUI.py`

which reads `out.txt` and writes visualization images to a folder called `visual` in the root directory. 

## Optimization
This implementation provides several optimzation methods to speed up image retrieval. The first method is DOR (Dynamically Optimized Retrieval), which works by only considering the nearest particles and assigning small, non-zero probabilities to the other particles. This method is run using by calling

`>> analyzer.optP()`

instead of creating the raw output file and processing the raw output file. This also writes to `out.txt`. 

Another method is Bag-of-Words, which uses machine learning and natural language processing techniques to speed up image retrieval. Features are extracted from an image in the map, and clustered using k-means clustering. Clusters are transformed into vectors using tf-idf vectorization and stored as visual words in a dictionary. Each location in the map has a dictionary associated with it, stored in a file with the extension `.pkl` in the `map` folder of the root directory. 

To use Bag-of-Words, initialize the dictionaries.
```
python -i Matcher.py
>> matcher = Matcher('BOW')
>> matcher.writeIndices()
```
Next, run 
```
python -i analyze.py
>> analyzer = analyzer('BOW',320,240)
```
and follow the above steps outlined in the above section. Note that one cannot combine DOR and BOW, as they are mutually exclusive.

## Credits
Harvey Mudd College Computer Science REU

Ciante Jones: cjjones@hmc.edu

Chi-Yen Lee: johlee@hmc.edu

Andy Zhang: axzhang@hmc.edu
