<div align="center">
  <h1>PepT3</h1>
  <p><strong>Test-time training for deep MS/MS spectrum prediction improves peptide identification</strong></p>
  <p>
    <img src="https://github.com/gusye1234/PepT3/actions/workflows/main.yml/badge.svg">
  </p>
</div>


## Get Started in CLI

Run

```shell
python -m pept3 ./examples/demo_data/demo_input.tab --spmodel=prosit --similarity=SA --output_tab=./examples/demo_data/demo_out.tab --need_tensor --output_tensor=./examples/demo_data/tensor.hdf5
```
to perform a simple test-time training over Prosit(`--spmodel=prosit`) with Spectral Angle(`--similarity=SA`). 
The program will take `./examples/demo_data/demo_input.tab` as the input file. Then the tuned features will be outputted to `./examples/demo_data/demo_out.tab`, which is already for the downstream task, for example, as the input of the [Percolator](https://github.com/percolator/percolator):
```shell
cd examples
bash ./percolator.sh # rescoring over the tuned features set
# the result will be saved in ./examples/percolator_result
```

### Get Started in `PepT3` Internals

Run

```shell
cd examples
python pept3_demo.py
```

You should get the identical result as the *Get Started in CLI* section.
The script `pept3_demo.py` will demonstrate the process of how `PepT3` working inside python.

## Input Format

PepT3 expect a tab-delimited file format as the input, just like [Percolator](https://github.com/percolator/percolator/wiki/Interface#pintsv-tab-delimited-file-format).
Each row should contains features associated with a single PSM:

```
SpecId <tab> Label <tab> ScanNr <tab> matched_ions <tab> matched_inten <tab> ... Charge <tab> <tab> Peptide <tab>
```

For PepT3, the input tab file should at least include those fields:

* `SpecId`(any type): **Unique** id for each PSM.
* `ScanNr`(any type): Same meaning as the [Percolator](https://github.com/percolator/percolator/wiki/Interface#pintsv-tab-delimited-file-format).
* `Label`({1, -1}): 1 for target PSM, -1 for decoys.
* `matched_ions`: `;`-delimited matched ions for PSM, only b/y types are considered currently. For example `b10;b2;b3;`
* `matched_inten`: Corresponding ions' intensities for the matched ions, also `;`-delimited. For example `829;4154;168;`
* `Charge`(int):, Percursor Charge
* `collision_energy_aligned_normed`(float, [0,1]): Maximun-normalized NCE.
* `Peptides`(str)

For the input example, have a look at `./examples/demo_data/demo_input.tab`.
*Please note that: For any feature that not on the above list, PepT3 will automaticly merge it into the output tab*

## Output Format

PepT3 outputs a tab-delimited file format with each row contains enlarged features associated with a single PSM. The output tab file can be directly used as the input of the [Percolator](https://github.com/percolator/percolator). Have a look at each features meaning in `./FEATURES.txt`.
Also, for those who want to visit the tuned spectrum prediction, use `--need_tensor` option and set `--output_tensor`. The prediction will be store as the format of `hfd5`, with columns `SpecId` and `tuned-tensor`.
For the output example, please have a look at `./examples/demo_data/demo_out.tab` and `./examples/demo_data/tensor.hdf5`
