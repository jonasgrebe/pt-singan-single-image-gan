from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.shortcuts import render
from .forms import *
from .tasks import *
from celery.result import AsyncResult
import json
import pathlib

# Variables only accessible by Django
train_image_model = None
input_image_model = None
image_paths = None
singan_info = {}

modes_w_input_image = ['Super Resolution', 'Paint-to-Image', 'Harmonization', 'Injection']


def index(request):
    global image_paths
    image_paths = get_images_of_pretrained_models()
    return show_welcome(request, image_paths)


def show_welcome(request, image_paths):
    form = TrainingForm()
    return HttpResponse(render(request, 'singan_web/welcome.html', {'form': form, 'image_paths': image_paths}))


def post_training_form(request):
    global train_image_model, input_image_model

    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = TrainingForm(request.POST, request.FILES)

        if train_image_model is None:
            error_msg = "Please provide a training image."
            return render(request, 'singan_web/welcome.html', {'form': form, 'error_msg': error_msg})

        train_image_path = train_image_model.image.path

        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            N = int(form.cleaned_data['scales']) - 1
            r = float(form.cleaned_data['rescale_factor'])
            steps_per_scale = int(form.cleaned_data['steps_per_scale'])

            global singan_info
            singan_info = {'N': N, 'r': r, 'train_image_path': train_image_path}
            input_image_model = train_image_model

            # Execute asynchronous task in background
            # Run "celery -A web worker --pool=solo -l info" before execution
            # Kill all queued tasks with celery -A web purge
            training_task = train_singan.delay(train_image_path, N, r, steps_per_scale)
            print("Started task", training_task)

            # Meanwhile, redirect to new URL
            return render(request, 'singan_web/processing.html',
                          {'image_path': train_image_model.image.url, 'task_id': training_task.task_id})

        print("Got invalid image submit.")
    # if a GET (or any other method) we'll create a blank form
    else:
        form = TrainingForm()

    return render(request, 'singan_web/welcome.html', {'form': form})


def post_training_image(request):
    image = request.FILES['training_image']
    image_model = ImageModel()
    image_model.image = image
    image_model.save()

    global train_image_model
    train_image_model = image_model

    return HttpResponse()


def post_input_image(request):
    image = request.FILES['input_image']
    image_model = ImageModel()
    image_model.image = image
    image_model.save()

    global input_image_model
    input_image_model = image_model

    return HttpResponse()


def load_pretrained(request, model_id):
    global singan_info, image_paths, input_image_model

    model_id = int(model_id)

    singan_info = get_singan_info(model_id)

    image_model = ImageModel()
    image_media_path = pathlib.Path(image_paths[model_id])
    image_path = pathlib.Path(*image_media_path.parts[1:])
    image_model.image = str(image_path)
    input_image_model = image_model

    load_singan.delay(model_id)
    return show_generate(request)


def show_processing_page(request):
    template = loader.get_template('singan_web/processing.html')
    return HttpResponse(template.render({}, request))


def get_task_info(request, task_id):
    print("Got task info request.")

    if task_id is not None:
        task = AsyncResult(task_id)

        data = {
            'state': task.state,
            'result': task.result,
        }

        print("Responding...")
        return HttpResponse(json.dumps(data), content_type='application/json')
    else:
        print("No job id given.")
        return HttpResponse('No job id given.')


def cancel_training(request, task_id):
    task = AsyncResult(task_id)
    task.revoke(terminate=True)
    return HttpResponseRedirect(redirect_to="/")


def show_generate(request):
    template = loader.get_template('singan_web/generate.html')
    return HttpResponse(template.render({'mode': 'Select a mode'}, request))


def show_random_gen_form(request):
    task_id = None
    if request.method == 'POST':
        form = RandomGenerationForm(request.POST, request.FILES)
        if form.is_valid():
            res_x = int(form.cleaned_data['res_x'])
            res_y = int(form.cleaned_data['res_y'])

            async_task = generate_random_image.delay(res_x, res_y)
            task_id = async_task.task_id
    else:
        form = RandomGenerationForm()

    mode = 'Random Generation'
    description = 'This mode can be used to generate images randomly, ' \
                  'looking similar to the input image. The level of ' \
                  'detail where variation should happen can be selected.'
    show_image_upload = mode in modes_w_input_image
    return render(request, 'singan_web/generate.html', {'mode': mode,
                                                        'description': description,
                                                        'form': form,
                                                        'task_id': task_id,
                                                        'show_image_upload': show_image_upload,
                                                        'input_image_url': get_input_image_url()})


def show_super_res_form(request):
    task_id = None
    error_msg = None
    if request.method == 'POST':
        form = SuperResForm(request.POST, request.FILES)

        global input_image_model
        if input_image_model is None:
            error_msg = "Please provide an input image."
        elif form.is_valid():
            upscale_factor = int(form.cleaned_data['upscale_factor'])
            image_path = input_image_model.image.path
            async_task = generate_super_res.delay(image_path, upscale_factor)
            task_id = async_task.task_id
    else:
        form = SuperResForm()

    mode = 'Super Resolution'
    description = 'This mode can be used to increase the resolution ' \
                  'of the input image, adding finer details and higher ' \
                  'sharpness. Beware that too much super resolution ' \
                  'creates undesired artifacts.'
    show_image_upload = mode in modes_w_input_image
    return render(request, 'singan_web/generate.html', {'mode': mode,
                                                        'description': description,
                                                        'form': form,
                                                        'task_id': task_id,
                                                        'show_image_upload': show_image_upload,
                                                        'input_image_url': get_input_image_url(),
                                                        'error_msg': error_msg})


def show_paint2image_form(request):
    task_id = None
    error_msg = None
    if request.method == 'POST':
        form = Paint2ImageForm(singan_info['N'], request.POST, request.FILES)

        global input_image_model
        if input_image_model is None:
            error_msg = "Please provide an input image."
        elif form.is_valid():
            start_at_scale = int(form.cleaned_data['start_at_scale'])
            image_path = input_image_model.image.path

            async_task = generate_paint2image.delay(image_path, start_at_scale)
            task_id = async_task.task_id
    else:
        form = Paint2ImageForm(N=singan_info['N'])

    mode = 'Paint-to-Image'
    description = 'Here you can upload a hand-painted image consisting ' \
                  'of simple colors. SinGAN will then convert this ' \
                  'simple image into a real-looking image based on the ' \
                  'one it was trained on.'
    show_image_upload = mode in modes_w_input_image
    return render(request, 'singan_web/generate.html', {'mode': mode,
                                                        'description': description,
                                                        'form': form,
                                                        'task_id': task_id,
                                                        'show_image_upload': show_image_upload,
                                                        'input_image_url': get_input_image_url(),
                                                        'error_msg': error_msg})


def show_harmonization_form(request):
    task_id = None
    error_msg = None
    if request.method == 'POST':
        # form = HarmonizationForm(request.POST, request.FILES)

        global input_image_model
        if input_image_model is None:
            error_msg = "Please provide an input image."
        else:
            image_path = input_image_model.image.path
            async_task = generate_harmonization.delay(image_path)
            task_id = async_task.task_id

    mode = 'Harmonization'
    description = 'Using this mode you can make an input image look ' \
                  'more harmonic, i.e., back- and foreground seem as ' \
                  'they belong to the same image and are not cut-and' \
                  '-pasted together.'
    show_image_upload = mode in modes_w_input_image
    return render(request, 'singan_web/generate.html', {'mode': mode,
                                                        'description': description,
                                                        'task_id': task_id,
                                                        'show_image_upload': show_image_upload,
                                                        'input_image_url': get_input_image_url(),
                                                        'error_msg': error_msg})


def show_injection_form(request):
    global input_image_model
    task_id = None
    error_msg = None
    if request.method == 'POST':
        form = InjectionForm(singan_info['N'], request.POST, request.FILES)

        if input_image_model is None:
            error_msg = "Please provide an input image."
        elif form.is_valid():
            start_at_scale = int(form.cleaned_data['start_at_scale'])
            image_path = input_image_model.image.path
            async_task = generate_injection.delay(image_path, start_at_scale)
            task_id = async_task.task_id
    else:
        form = InjectionForm(N=singan_info['N'])

    mode = 'Injection'
    description = 'This mode enables you to manipulate a given image ' \
                  '(not necessarily the one SinGAN was trained on) on ' \
                  'a specified level of detail.'
    show_image_upload = mode in modes_w_input_image
    return render(request, 'singan_web/generate.html', {'mode': mode,
                                                        'description': description,
                                                        'form': form,
                                                        'task_id': task_id,
                                                        'show_image_upload': show_image_upload,
                                                        'input_image_url': get_input_image_url(),
                                                        'error_msg': error_msg})


def get_input_image_url():
    global input_image_model
    input_image_url = None
    if input_image_model is not None:
        input_image_url = input_image_model.image.url
    return input_image_url
