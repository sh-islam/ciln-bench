# data/softmaxes (not in git)

`fig02_dists.py` (Figs 2 and 7) and `fig08_appendix.py` (Figs 8 and 9) read raw voter outputs from this folder. They are the released files, so they are not duplicated here. Rebuild the folder from the Hugging Face datasets with this layout:

```
<dataset>/<corruption>/severity_<l>/noisy_label_train/{labels.npy, softmax_<voter>.npy}
clean/<dataset>/noisylabeltrain_clean/labels.npy
```

Folders the scripts expect:

```
  adult
  adult/gaussian_noise
  adult/gaussian_noise/severity_1
  adult/gaussian_noise/severity_3
  adult/gaussian_noise/severity_5
  adult/missing_mar
  adult/missing_mar/severity_1
  adult/missing_mar/severity_3
  adult/missing_mar/severity_5
  adult/missing_mcar
  adult/missing_mcar/severity_1
  adult/missing_mcar/severity_3
  adult/missing_mcar/severity_5
  adult/missing_mnar
  adult/missing_mnar/severity_1
  adult/missing_mnar/severity_3
  adult/missing_mnar/severity_5
  adult/scaling
  adult/scaling/severity_1
  adult/scaling/severity_3
  adult/scaling/severity_5
  agnews
  agnews/butter_fingers
  agnews/butter_fingers/severity_1
  agnews/butter_fingers/severity_3
  agnews/butter_fingers/severity_5
  agnews/front_truncation
  agnews/front_truncation/severity_1
  agnews/front_truncation/severity_3
  agnews/front_truncation/severity_5
  cifar10
  cifar10/brightness
  cifar10/brightness/severity_1
  cifar10/brightness/severity_3
  cifar10/brightness/severity_5
  cifar10/gaussian_noise
  cifar10/gaussian_noise/severity_1
  cifar10/gaussian_noise/severity_3
  cifar10/gaussian_noise/severity_5
  cifar10/pixelate
  cifar10/pixelate/severity_1
  cifar10/pixelate/severity_3
  cifar10/pixelate/severity_5
  clean
  clean/adult
  clean/adult/noisylabeltrain_clean
  clean/agnews
  clean/agnews/noisylabeltrain_clean
  clean/cifar10
  clean/cifar10/noisylabeltrain_clean
  clean/mnist
  clean/mnist/noisylabeltrain_clean
  mnist
  mnist/brightness
  mnist/brightness/severity_1
  mnist/brightness/severity_3
  mnist/brightness/severity_5
  mnist/rotate
  mnist/rotate/severity_1
  mnist/rotate/severity_3
```

`labels.npy` and `softmax_<voter>.npy` come from `settings/<corruption>_sev<l>/noisy_label_train/` in each dataset repo. The clean labels come from `clean_start/labels.npy`.
