* Retained owned GPU annotator frames through their tiled-image copy stream fence,
  including exceptional exits, before returning camera outputs or destroying
  render products. CPU image readback and image layout were unchanged.
