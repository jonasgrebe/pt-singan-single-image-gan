from django.http import HttpResponse
from django.template import loader


def index(request):
    template = loader.get_template('singan_web/welcome.html')
    # return render_to_response('singan_web/welcome.html', {})
    return HttpResponse(template.render({}, request))
