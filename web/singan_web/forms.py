from django import forms

LEVELS_OF_DETAIL = (
    ("1", "Tiny details"),
    ("2", "Medium details & textures"),
    ("3", "Small objects"),
    ("4", "Medium objects and parts"),
    ("5", "Large objects and areas"),
    ("6", "Whole image composition"),
)


class TrainingForm(forms.Form):
    user_image = forms.ImageField(label="")

    scales = forms.DecimalField(label="Number of scales (N)", min_value=0, max_value=12, initial=6)
    steps_per_scale = forms.DecimalField(label="Steps per scale", min_value=1, max_value=100000, initial=2000)
    rescale_factor = forms.DecimalField(label="Rescale factor", min_value=1, max_value=10, initial=1.33)


class RandomGenerationForm(forms.Form):
    insert_stage_id = forms.ChoiceField(label='Level of detail', choices=LEVELS_OF_DETAIL)

    res_x = forms.DecimalField(label="Width")
    res_y = forms.DecimalField(label="Height")


class SuperResForm(forms.Form):
    user_image = forms.ImageField(label="")

    upscale_factor = forms.ChoiceField(label="", choices=(
        ("1", "Slightly higher resolution"),
        ("2", "Remarkably higher resolution"),
        ("3", "Strongly higher resolution"),
        ("4", "Excessively higher resolution"),
        ("5", "Garbage. Don't use this"),
    ))


class Paint2ImageForm(forms.Form):
    painted_image = forms.ImageField(label="Input image")

    insert_stage_id = forms.ChoiceField(label='Level of detail', choices=LEVELS_OF_DETAIL)


class HarmonizationForm(forms.Form):
    image_to_harmonize = forms.ImageField(label="Input image")
