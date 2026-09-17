# Copied verbatim from the user's MobileCLIP2 quick-infer branch (f1cc914).
LEGACY_PROMPTS = {
    'night': (['a real photograph of a scene at nighttime', 'a night scene photographed after sunset'],
              ['a real photograph of a scene in daytime', 'an indoor scene in ordinary lighting, not a night scene']),
    'indoor': (['a photograph taken inside a room or building', 'an indoor environment'],
               ['a photograph taken in an outdoor environment', 'an open-air exterior scene']),
    'rain_snow': (['a photograph with visible rain or snow', 'a scene showing rainfall, snowfall, or snow on the ground'],
                  ['a photograph with neither rain nor snow present', 'a scene without rain and without snow']),
    'office': (['a photograph of an office workspace or office meeting room', 'a professional office, cubicle, or home office'],
               ['a scene that is not an office workspace', 'a non-office environment']),
    'outdoor': (['a photograph taken in an outdoor environment', 'an open-air exterior scene'],
                ['a photograph taken inside a room or building', 'an indoor environment']),
    'landscape': (['a photograph whose main subject is a natural landscape', 'natural scenery, mountains, forest, or sea as the main subject'],
                  ['a photograph whose main subject is a person, animal, object, or city', 'a cityscape or close-up subject rather than natural scenery']),
    'sports': (['a photograph of sports activity or a sporting event', 'a sports venue, court, playing field, or people doing sports'],
               ['a scene unrelated to sports', 'a non-sports scene or activity']),
    'objective_image': (['an image quality test pattern or resolution chart', 'a computer-generated calibration pattern, color bars, or test chart'],
                        ['a real photograph of a scene rather than an image quality test pattern', 'a real office or room containing monitors, not a test chart image']),
}
