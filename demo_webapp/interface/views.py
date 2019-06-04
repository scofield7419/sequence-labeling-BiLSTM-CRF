from django.shortcuts import render
from django.views.decorators.csrf import csrf_protect
from django.http import HttpResponse
import json, random
from interface import model, logger, configs

if configs.label_level == 1:
    from interface import color_list
elif configs.label_level == 2:
    from interface import color_dict


def index(request):
    return render(request, 'index.html')


def api(request):
    sentence = request.GET["sentence"]
    sentence_tokens, entities, entities_type, entities_index = model.predict_single(sentence)
    if len(entities) != 0:
        if configs.label_level ==1:
            respo = {
                "sentence": ' '.join(sentence_tokens),
                "entities": entities,
            }
        elif configs.label_level ==2:
            respo = {
                "sentence": ' '.join(sentence_tokens),
                "entities": [{ent: typ} for ent, typ in zip(entities, entities_type)],
            }
    else:
        respo = {
            "sentence": sentence,
            "entities": '',
        }
    return HttpResponse(json.dumps(respo), content_type="application/json")


@csrf_protect
def predict(request):
    sentence = request.POST["sentence"].strip()
    logger.info(sentence)

    if sentence.strip() != "":
        sentence_tokens, entities, entities_type, entities_index = model.predict_single(sentence)

        logger.info("\nSentence tokens:\n %s\n" % (" ".join(sentence_tokens)))
        if configs.label_level == 1:
            logger.info("Extracted entities:\n %s\n" % ("\n".join(
            [ent + "\t([%d-%d])" % (inda, indb) for ent, (inda, indb) in
             zip(entities, entities_index)])))
        elif configs.label_level == 2:
            logger.info("Extracted entities:\n %s\n" % ("\n".join(
            [ent + "\t(%s;[%d-%d])" % (typ, inda, indb) for ent, typ, (inda, indb) in
             zip(entities, entities_type, entities_index)])))


        html_str_ = r'<h3 style="font-weight: bold;line-height: 42px;text-indent:3em;margin-top: 20px" class="resh2">%s</h3>'
        html_str_inner_level1 = r'<mark style="background-color:#%s" >%s</mark>'
        html_str_inner_level2 = r'<mark style="background-color:#%s" >%s<sub style="font-weight: normal;font-size: small"> (%s)</sub></mark>'

        text_ = ''
        for i, token in enumerate(sentence_tokens):
            if len(entities_index) > 0:
                if entities_index[0][0] <= i and i < entities_index[0][1]:
                    continue

            if len(entities_index) > 0 and i == entities_index[0][1]:
                if configs.label_level == 1:
                    text_ += (" " + html_str_inner_level1 % (random.choice(color_list), entities[0]))
                elif configs.label_level == 2:
                    text_ += (" " + html_str_inner_level2 % (color_dict[entities_type[0]], entities[0], entities_type[0]))

                entities_index = entities_index[1:]
                entities_type = entities_type[1:]
                entities = entities[1:]
            text_ += (" " + token)

        json_data = {'info': html_str_ % text_}
    else:
        json_data = {'info': ""}

    logger.info(json_data)

    return HttpResponse(json.dumps(json_data), content_type="application/json")
