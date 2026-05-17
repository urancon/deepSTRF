# CRCNS MT1

**Dataset source**: [CRCNS MT1](https://crcns.org/data-sets/vc/mt-1)

**Citation**: 
```text
Cui Y, Liu DL, Khawaja FA, Pack CC, Butts DA (2013).  Spiking activity in area MT of awake adult macaques in response 
to complex motion features. CRCNS.org.  http://dx.doi.org/10.6080/K0X63JTX
```

**Associated papers**:
- ["Diverse suppressive influences in area MT and selectivity to complex motion features"](https://www.jneurosci.org/content/33/42/16715.long)
  (2013), Cui et al., J Neurosci

  
# Dataset details

**Population fitting:** ❌

**Description of Stimuli:**
- kinematograms featuring complex optic flow patterns
- aperture with slowly moving position across the visual field
- long sequences, variable length across neurons
- dt @ XXX
- only 1 response trial

**Description of Neurons:**
- Extracellular single-unit recordings from 2 macaque monkeys
- MT area
- Total Number of Neurons: 84

**Available data:**
- originally stored in Matlab (.mat) format (one file per cell)
- Matlab & Python preprocessing.
- need to generate stimuli from hyperparameters (optic flow components)
- 5 sec repeated stimuli (cf. original article) not shared by the authors



# Setup Instructions

**Requirements:** a [CRCNS](https://crcns.org/) account, [Matlab](https://www.mathworks.com/products/matlab.html) or alternatively [Octave](https://octave.org)
0. Download the data (**crcns-mt1.zip**) at [the original dataset repository](https://crcns.org/data-sets/vc/mt-1/about) and unzip it.
2. Run the script `preprocessing_scripts/generate_crncs_mt1_derived.m` to extract the data. 
Don't forget to set the correct paths before execution.
3. Run the script `preprocessing_scripts/generate_crcns_mt1_derived.py` to generate the optic flow stimuli. 
Don't forget to set the correct paths before execution.
4. Move all `movie<idx>.h5` files to the `data/` folder. You should have the following directory structure:
```text


```
5. You can use it right away by creating a `CRCNS_MT1_Dataset(...)` object with the path to the **data/** folder
