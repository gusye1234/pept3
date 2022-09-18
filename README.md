<div align="center">
  <h1>TFP</h1>
  <p><strong>Data-specific model fine-tuning for MS/MS spectrum prediction</strong></p>
  <p>
    <img src="https://github.com/gusye1234/TFP/actions/workflows/main.yml/badge.svg">
  </p>
</div>


## Get Started in CLI

Run

```shell
python -m tfpms ./examples/demo_data/demo_input.tab --spmodel=prosit --similarity=SA --output_tab=./examples/demo_data/demo_out.tab --need_tensor --output_tensor=./examples/demo_data/tensor.hdf5
```

to perform a simple fine-tuneing over Prosit(`--spmodel=prosit`) with Spectral Angle(`--similarity=SA`). The program will take `./examples/demo_data/demo_input.tab` as the input file.

Then the finetuned features is outputed to `./examples/demo_data/demo_out.tab`, which is already for the downstream task, for example, as the input of the [Percolator](https://github.com/percolator/percolator):

```shell
cd examples
bash ./percolator.sh # rescoring over the fine-tuned features set
# the result will be saved in ./examples/percolator_result
```

### Get Started in `TFP` Internals

Run

```shell
cd examples
python finetune_demo.py
```

You should get the identical result as the *Get Started in CLI* section.
The script `finetune_demo.py` will demonstrate the process of how `TFP` working inside python.

## Input Format

TFP expect a tab-delimited file format as the input, just like [Percolator](https://github.com/percolator/percolator/wiki/Interface#pintsv-tab-delimited-file-format).
Each row should contains features associated with a single PSM:

```
SpecId <tab> Label <tab> ScanNr <tab> matched_ions <tab> matched_inten <tab> ... Charge <tab> <tab> Peptide <tab>
```

For TFP, the input tab file should at least include those fields:

* `SpecId`(any type): **Unique** id for each PSM.
* `ScanNr`(any type): Same meaning as the [Percolator](https://github.com/percolator/percolator/wiki/Interface#pintsv-tab-delimited-file-format).
* `Label`({1, -1}): 1 for target PSM, -1 for decoys.
* `matched_ions`: `;`-delimited matched ions for PSM, only b/y types are considered currently. For example `b10;b2;b3;`
* `matched_inten`: Corresponding ions' intensities for the matched ions, also `;`-delimited. For example `829;4154;168;`
* `Charge`(int):, Percursor Charge
* `collision_energy_aligned_normed`(float, [0,1]): Maximun-normalized NCE.
* `Peptides`(str)

For the input example, have a look at `./examples/demo_data/demo_input.tab`.
*Please note that: For any feature that not on the above list, TFP will automaticly merge it into the output tab*

## Output Format

TFP outputs a tab-delimited file format with each row contains enlarged features associated with a single PSM. The output tab file can be directly used as the input of the [Percolator](https://github.com/percolator/percolator). Have a look at each features meaning in `./FEATURES.txt`.
Also, for those who want to visit the fine-tuned spectrum prediction, use `--need_tensor` option and set `--output_tensor`. The prediction will be store as the format of `hfd5`, with columns `SpecId` and `fine-tuned-tensor`.
For the output example, please have a look at `./examples/demo_data/demo_out.tab` and `./examples/demo_data/tensor.hdf5`
