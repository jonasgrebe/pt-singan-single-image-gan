from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
from .forms import *
from .models import ImageModel
from .tasks import *
from celery.result import AsyncResult
import json


def index(request):
    return HttpResponse(get_training_form(request))


def get_training_form(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = TrainingForm(request.POST, request.FILES)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            image = request.FILES['user_image']
            image_model = ImageModel()
            image_model.image = image
            image_model.save()
            print("Saved uploaded image.")

            image_path = image_model.image.path
            N = int(form.cleaned_data['scales'])
            r = float(form.cleaned_data['rescale_factor'])
            steps_per_scale = int(form.cleaned_data['steps_per_scale'])

            # Execute asynchronous task in background
            # Run "celery -A web worker --pool=solo -l info" before execution
            # Kill all queued tasks with celery -A web purge
            training_task = train_singan.delay(image_path, N, r, steps_per_scale)
            print("Started task", training_task)

            # Meanwhile, redirect to new URL
            return render(request, 'singan_web/processing.html',
                          {'image_model': image_model, 'task_id': training_task.task_id})

        print("Got invalid image submit.")
    # if a GET (or any other method) we'll create a blank form
    else:
        form = TrainingForm()

    return render(request, 'singan_web/welcome.html', {'form': form})


def load_singan(request):
    load_singan().delay()


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


def show_generate(request):
    template = loader.get_template('singan_web/generate.html')
    return HttpResponse(template.render({'mode': 'Select a mode'}, request))


def show_random_gen_form(request):
    task_id = None
    if request.method == 'POST':
        form = RandomGenerationForm(request.POST, request.FILES)
        if form.is_valid():
            insert_stage_id = int(form.cleaned_data['insert_stage_id'])
            res_x = int(form.cleaned_data['res_x'])
            res_y = int(form.cleaned_data['res_y'])

            async_task = generate_random_image.delay(insert_stage_id, res_x, res_y)
            task_id = async_task.task_id

    mode = 'Random Generation'
    description = 'This mode can be used to generate images randomly, ' \
                  'looking similar to the input image. The level of ' \
                  'detail where variation should happen can be selected.'
    form = RandomGenerationForm()
    return render(request, 'singan_web/generate.html', {'mode': mode,
                                                        'description': description,
                                                        'form': form,
                                                        'task_id': task_id})


def show_super_res_form(request):
    task_id = None
    if request.method == 'POST':
        form = SuperResForm(request.POST, request.FILES)
        task_id = run_singan(form)

    mode = 'Super Resolution'
    description = 'This mode can be used to increase the resolution ' \
                  'of the input image, adding finer details and higher ' \
                  'sharpness. Beware that too much super resolution ' \
                  'creates undesired artifacts.'
    form = SuperResForm()
    return render(request, 'singan_web/generate.html', {'mode': mode,
                                                        'description': description,
                                                        'form': form,
                                                        'task_id': task_id})


def show_paint2image_form(request):
    task_id = None
    if request.method == 'POST':
        form = Paint2ImageForm(request.POST, request.FILES)
        task_id = run_singan(form)

    mode = 'Paint-to-Image'
    description = 'Here you can upload a hand-painted image consisting ' \
                  'of simple colors. SinGAN will then convert this ' \
                  'simple image into a real-looking image based on the ' \
                  'one it was trained on.'
    form = Paint2ImageForm()
    return render(request, 'singan_web/generate.html', {'mode': mode,
                                                        'description': description,
                                                        'form': form,
                                                        'task_id': task_id})


def show_harmonization_form(request):
    task_id = None
    if request.method == 'POST':
        form = HarmonizationForm(request.POST, request.FILES)
        task_id = run_singan(form)

    mode = 'Harmonization'
    description = 'With this mode you can make an input image look ' \
                  'more harmonic, i.e., back- and foreground look as ' \
                  'they belong to the same image and are not cut-and' \
                  '-pasted together.'
    form = HarmonizationForm()
    return render(request, 'singan_web/generate.html', {'mode': mode,
                                                        'description': description,
                                                        'form': form,
                                                        'task_id': task_id})


def run_singan(form):
    print("Received form.")

    task_id = None
    if form.is_valid():
        # Execute asynchronous task in background
        # Run "celery -A web worker --pool=solo -l info" before execution
        async_task = my_task.delay(4)
        task_id = async_task.task_id
        print("Started SinGAN computation task", async_task)
    else:
        print("Got invalid form submit.")

    return task_id
