from django import forms

LEVELS_OF_DETAIL = (
    ("1", "Tiny details"),
    ("2", "Medium details & textures"),
    ("3", "Small objects"),
    ("4", "Medium objects and parts"),
    ("5", "Large objects and areas"),
    ("6", "Whole image composition"),
)


# decimal_widget = forms.NumberInput(attrs={'class': 'form-control'})


class TrainingForm(forms.Form):
    scales = forms.DecimalField(label="Number of scales (N)", min_value=0, max_value=12, initial=6, decimal_places=0)
    steps_per_scale = forms.DecimalField(label="Steps per scale", min_value=1, max_value=100000, initial=2000,
                                         decimal_places=0)
    rescale_factor = forms.DecimalField(label="Rescale factor", min_value=1, max_value=10, initial=1.33)
    # training_image = forms.ImageField(label="Training image")


class RandomGenerationForm(forms.Form):
    res_x = forms.DecimalField(label="Width", min_value=1, max_value=2000, initial=600)
    res_y = forms.DecimalField(label="Height", min_value=1, max_value=2000, initial=450)


class SuperResForm(forms.Form):
    upscale_factor = forms.ChoiceField(label="", choices=(
        ("1", "Slightly higher resolution"),
        ("2", "Remarkably higher resolution"),
        ("3", "Strongly higher resolution"),
        ("4", "Excessively higher resolution"),
        ("5", "Garbage. Don't use this"),
    ))


class Paint2ImageForm(forms.Form):
    def __init__(self, N, *args, **kwargs):
        super(Paint2ImageForm, self).__init__(*args, **kwargs)

        levels_of_detail = list(zip(range(N), get_detail_labels(N)))

        self.fields['start_at_scale'] = \
            forms.ChoiceField(label='Level of detail',
                              choices=levels_of_detail,
                              initial=str(N))


class InjectionForm(forms.Form):
    def __init__(self, N, *args, **kwargs):
        super(InjectionForm, self).__init__(*args, **kwargs)

        levels_of_detail = list(zip(range(N), get_detail_labels(N)))

        self.fields['start_at_scale'] = \
            forms.ChoiceField(label='Level of detail',
                              choices=levels_of_detail,
                              initial=str(N))


def get_detail_labels(n):
    labels = list(range(1, n+1))
    labels[0] = "1: Tiny details"
    labels[-1] = "%d: Whole image composition" % len(labels)
    return labels
