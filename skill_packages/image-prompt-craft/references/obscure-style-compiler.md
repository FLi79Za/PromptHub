# Obscure-style compiler

Use this reference only when a user asks for an obscure, historical, technical, regional, photographic, scientific, printmaking, vernacular, experimental or invented visual style, or when a familiar style name is unlikely to be understood reliably by the target image model.

## Purpose

Compile:

`style name or aesthetic concept → canonical visual profile → model-specific translation → final prompt`

The style name is a lookup hint, not the prompt. The usable output is a compact set of visible, material and process behaviours that a model can render.

## Classification and confidence

Classify the request before compiling it:

- **Common**: the name has a stable, widely shared visual meaning. Use the name as a useful anchor, then add only the traits that matter.
- **Partially understood**: the name has several legitimate meanings, or models commonly confuse it with a neighbouring style. State the chosen interpretation briefly and include disambiguating traits.
- **Obscure/technical**: the name refers to a process, apparatus, regional practice or historical medium. Translate the process into visible consequences and retain the name only as a secondary anchor.
- **Hybrid or invented**: the user combines incompatible traditions or has coined a term. Preserve the intended combination, separate its component signatures, and state the deliberate synthesis when useful.

Use confidence labels internally: **high**, **moderate** or **uncertain**. Do not present a disputed attribution, date, regional association, chemical explanation or historical claim as settled fact. If the source material supplies a striking but weakly supported claim, use the visual signature only and avoid repeating the claim as fact.

## Canonical visual profile

Build the profile in this order. Omit fields that do not affect the image.

1. **Medium and substrate**: paper, silvered plate, blackened iron, glass, pigment film, textile, wall, screen, film emulsion, transparent overlay or digital raster.
2. **Image formation or mark-making**: contact exposure, intaglio, relief cut, stencil, layered pigment, chemical bleaching, developer staining, corona discharge, refraction, scanline, collage, hand lettering or another physical operation.
3. **Palette and tonal structure**: dominant and secondary colours, temperature, saturation, contrast, density of blacks, highlight behaviour and whether colour is continuous, granular, misregistered or separated into plates.
4. **Optical properties**: focus, depth, lens distortion, pointillist grain, halation, iridescence, transparency, glow, refraction, motion blur or lack of camera perspective.
5. **Surface and material response**: paper tooth, plate sheen, gelatin relief, ink spread, gouge direction, pigment pooling, fibres, scratches, cracks, embossing, screen texture or glossy emulsion.
6. **Ageing and process artefacts**: fading, silver mirroring, edge fog, stains, light leaks, registration errors, reticulation, dust, uneven exposure, torn edges or chemical veils. Add artefacts only when they belong to the process, rather than as generic “vintage” decoration.
7. **Composition and conventions**: framing, viewpoint, negative space, repeated motifs, typography, scientific annotation, portrait pose, poster hierarchy, map logic or historical subject conventions.

Then separate **invariants** from **style variables**. In an edit, identity, anatomy, object geometry, layout, text, pose and requested setting may be invariants. Style variables include substrate, palette, edge behaviour, grain, marks, contrast and process defects.

## Translation procedure

1. Resolve the exact sense of the term. Ask one concise question only if two interpretations would produce materially different images; otherwise choose the most defensible reading and state it briefly.
2. Extract a canonical profile using the seven fields above.
3. Convert each field into observable language: “blue-white silhouettes embedded in matte paper” is more useful than “cyanotype” alone.
4. Select the target model reference already required by the main skill. Apply its syntax, emphasis, text-handling and editing rules after compiling the style.
5. Put the subject, action, composition and required text first where that model benefits from them. Put process signature and material cues next. Put restrained ageing and defects last.
6. For a hybrid, identify the dominant process, then add one or two controlled secondary signatures. Avoid stacking unrelated named styles.
7. For an edit or reference-guided task, use the main skill's `CHANGE`, `PRESERVE` and `MATCH` structure. Style conversion must not silently rewrite the subject or composition.
8. Do not append a generic negative prompt or quality-token list. If a failure mode is important, express it as a positive constraint, such as “single coherent exposure with deliberate three-colour registration offset”.

## Model-neutral compiler template

Use this as an internal scaffold, not as a mandatory visible heading structure:

`[subject and action], [composition and era convention], made as [medium/substrate] using [image-formation process], showing [palette and tonal behaviour], [optical behaviour], [surface texture], and [process-specific artefacts]. Preserve [invariants].`

Prefer two or three high-information process cues over a long catalogue of defects. A style should remain recognisable if the name is removed.

## Curated signature library

These are reusable signatures distilled from the supplied style studies. They are representative anchors, not an exhaustive dictionary.

### Alternative photographic processes

- **Mordançage**: silver-gelatin black-and-white print with acid-bleached highlights, lifted or veiled emulsion, branching chemical fissures, cracked relief and irregular dark-to-light transitions.
- **Physautotype**: very early silver-plate image with a pale, powdery, ghost-like deposit, low density, delicate detail and reflective plate presence. Treat historical chemistry and exact appearance as moderate confidence.
- **Gum bichromate**: soft-focus photograph built from translucent pigment-and-gum layers, uneven brush or coating edges, muted pictorial colour, paper tooth and selectively lost detail.
- **Carbon transfer**: matte pigmented image with continuous tonal modelling, dense dark areas that feel physically deposited, slight relief and visible paper texture.
- **Salt print**: matte fibrous paper, low to moderate contrast, warm sepia-brown silver image, softened edges and uneven hand-coated exposure.
- **Platinum/palladium**: broad quiet tonal scale, matte surface, luminous mid-tones, deep paper-embedded blacks and restrained neutral or warm grey-brown toning.
- **Cyanotype**: contact-print or blueprint logic, strong Prussian blue field, pale blue-white silhouettes or linework, matte paper and crisp exposure boundaries.
- **Cyanotype tri-colour**: cyanotype-like layered exposures in several colour channels, visibly imperfect registration, overlapping blue and complementary colour fringes. Treat as experimental rather than a single standard process.
- **Autochrome**: early colour photograph formed from fine coloured grain, pastel pointillist colour, soft detail, luminous highlights and a speckled plate texture.
- **Lippmann photography**: reflective plate-like image with structural, angle-dependent iridescent colour, restrained detail and a metallic surface response. Do not describe it as ordinary rainbow lighting.
- **Cibachrome/Ilfochrome**: positive print with unusually saturated clean colour, glossy surface, hard separation and deep opaque shadows.
- **Ambrotype**: detailed monochrome positive on glass read against a dark backing, dense blacks, glass sheen, edge fall-off and a one-off portrait-object quality.
- **Tintype/ferrotype**: monochrome portrait on darkened metal, central sharpness and contrast, metallic blacks, reflective highlights, uneven edges and small plate imperfections.
- **Anthotype**: plant-pigment image with highly faded organic pastel or near-monochrome colour, soft silhouettes and fragile uneven exposure.
- **Lumen print**: unfixed sun exposure on photographic paper, pale ghostly botanical or object traces, chemical discolouration, irregular bleaching and ephemeral low contrast.
- **Bromoil**: silver print bleached then reworked with ink, soft impressionistic masses, brushy or stippled ink texture, subdued tonal detail and a printmaking surface.
- **Chemigram**: camera-less photographic paper marked by resists, developer, fixer or varnish, branching stains, pools, halos, splashes and unpredictable positive-negative chemistry.
- **Rayograph/photogram**: camera-less contact exposure, crisp and soft object silhouettes, overlapping translucent shapes, exposure halos and a deliberate arrangement on light-sensitive paper.
- **Photogravure/heliogravure**: photographic subject carried by intaglio grain, etched or inked texture, rich darks, fine granular detail and slightly softened plate edges.
- **Solarisation/Sabattier**: partially reversed tonal relationships, outlined edges, metallic or pearlescent bands, abrupt local contrast and a darkroom accident made visible.
- **Lith print**: high-contrast graphic darkroom print, gritty infectious grain, harsh black masses, warm or split tones and unstable highlight edges.
- **Toy-camera/Holga or Lomography**: simple snapshot composition, strong vignette, lens softness, colour shifts, light leaks and saturated or cross-processed colour. Keep the lo-fi defects subordinate to the subject.

### Printmaking, illustration and ephemera

- **Mezzotint**: velvet-black intaglio field, smooth continuous half-tones, soft transitions and selectively scraped highlights rather than linear cross-hatching.
- **Aquatint**: granular etched tonal fields, smoky washes, restrained linework and plate-driven transitions from pale stain to dark tone.
- **Viscosity printing**: multi-colour intaglio image with separate ink viscosities creating controlled colour separation, soft plate texture and imperfect but intentional interaction between layers.
- **Pochoir**: hand-applied stencil colour, crisp bounded shapes, slight variation between colour areas, fine paper and decorative poster or book-plate composition.
- **Chromolithographic trade card**: bright layered commercial colour, small registration offsets, flat decorative shading, ornate border or lettering and period advertising hierarchy.
- **Chiaroscuro woodcut**: multiple relief blocks, bold cut contours, separated light and dark colour masses and visible carved grain.
- **Sosaku-hanga**: self-carved expressive gouges, rough wood texture, asymmetrical marks, direct personal gesture and a less polished surface than conventional ukiyo-e.
- **Stipple engraving**: image built from deliberate dots, controlled density for tonal modelling, crisp printed contours and a restrained monochrome or limited-colour palette.
- **Dada photomontage**: cut photographs, torn paper, mismatched scale, visible seams, pasted typography and deliberate visual dissonance.
- **Zine xerography**: photocopied black-and-white or limited-colour page, coarse toner grain, crushed shadows, registration drift, clipped text and torn or photocopied edges.
- **Letterpress woodtype poster**: heavy block lettering, ink squeeze, uneven pressure, paper impression, limited spot colours and simple hierarchical layout.

### Movements and regional or vernacular conventions

- **Rayonism**: intersecting rays and refracted planes, energetic colour wedges, broken spatial logic and a luminous abstract rhythm rather than literal beams in a realistic scene.
- **Orphism**: concentric discs, circular rhythms, prismatic colour relationships and overlapping translucent forms with a musical, non-narrative composition.
- **Vorticism**: angular compressed forms, hard-edged machine energy, fractured geometry, limited industrial palette and forceful directional movement.
- **Ashcan-style urban realism**: observed working-city streets, crowded irregular composition, blunt brushwork, dirty subdued colour and unidealised everyday activity. Avoid treating it as generic “gritty”.
- **Precisionism**: simplified industrial architecture, clean hard edges, reduced geometric planes, quiet atmosphere and controlled cool or neutral colour.
- **Hurufiyya**: Arabic letterforms treated as abstract visual structure, calligraphic rhythm, layered geometric fields and culturally appropriate script handling. Do not invent illegible pseudo-Arabic when actual text is required.
- **Zaum**: experimental visual-poetry layout, invented or fragmented language, phonetic marks, unusual spacing and page composition. Treat semantics and historical classification as uncertain; preserve the typographic experiment.
- **Rorke's Drift linocut**: use only when the user clearly intends the South African printmaking reference. Translate to bold linocut cuts, simplified figures, rhythmic repeated marks, flattened narrative space and integrated lettering. The supplied attribution and description are treated as uncertain, so do not assert a universal canonical style.
- **Persian miniature**: flattened multi-register space, fine contour, intricate pattern, jewel-like controlled colour and dense narrative detail. Preserve culturally specific costume and architecture when requested.
- **Japanese ukiyo-e or shin-hanga**: flat colour planes, deliberate contour, patterned surfaces, controlled perspective and a clear distinction between older ukiyo-e conventions and later shin-hanga refinement.
- **Wax-batik vernacular pattern**: resist-dyed cloth logic, repeated motifs, cracked wax boundaries, saturated layered colour and textile weave. Do not use generic “tribal” wording as a substitute for a named regional tradition.

### Scientific, technical and anomalous imaging

- **Schlieren photography**: monochrome or restrained colour laboratory image where refractive-index changes reveal airflow as sharp wave-like light and dark structures around an object. It is not smoke, fog or ordinary motion blur.
- **Kirlian/corona-discharge image**: dark photographic ground, luminous branching electrical corona hugging an object, high-voltage edge glow and contact-plate artefacts. Avoid claiming that it depicts a biological aura.
- **X-ray diffraction/crystallography**: central beam, radial or spot-based diffraction geometry, measured scientific composition, dark field and precise luminous traces.
- **X-ray film**: translucent radiographic density, anatomical or material penetration, cool grey-blue monochrome, soft exposure gradients and clipped high-density regions.
- **Electron micrograph**: extreme magnification, granular or fibrous microstructure, shallow apparent depth, instrument-like contrast and scientific framing rather than decorative macro photography.
- **MRI/CAT slice**: sectional medical imaging, anatomical slice logic, false-colour or clinical grayscale mapping, annotation space and consistent spatial orientation.
- **Astronomical false-colour composite**: nebular or planetary structure mapped into deliberate channel colours, deep black field, luminous gas boundaries and data-visualisation logic.
- **LIDAR point-cloud render**: sparse or dense spatial points encoding surfaces, depth-aware colour mapping, dark or transparent background and visible scan coverage rather than a conventional photograph.
- **Geological survey map**: contour lines, stratified colour regions, symbols, measured top-down geometry and restrained annotation hierarchy.
- **Histology slide**: thin-section microscopic field, stained cellular structures, translucent tissue, repeating biological forms and laboratory colour mapping.
- **Forensic or technical scan**: controlled neutral lighting, evidence-centred framing, scale references or annotation zones, texture revealed by imaging rather than cinematic drama.

### Screen, reproduction and experimental digital signatures

- **Retro CRT/VHS**: phosphor colour separation, scanlines, softened low-resolution image, interlacing or tracking disturbance, magnetic noise and restrained chromatic bleed.
- **Datamosh/video feedback**: block or motion-vector smearing, frame persistence, displaced colour fields and digital corruption that follows movement rather than random scratches.
- **Blueprint technical drawing**: white or pale cyan linework on a dense blue ground, orthographic or isometric construction, dimension marks, title block and disciplined spacing.
- **Soviet Constructivist poster**: diagonal geometry, restricted red-black-cream palette, bold sans-serif hierarchy, photomontage and directional propaganda composition. Avoid reproducing real political slogans unless requested.
- **Psychedelic poster**: vibrating complementary colours, warped lettering, flowing contour, dense pattern and optical rhythm. Keep text legible when the user requests readable text.
- **ASCII art**: image-forming character grid, discrete tonal density, monospaced alignment and visible text characters used as marks. Do not describe this as pixel art.
- **Paint-on-glass animation**: translucent pooled pigment, backlit colour, smeared edges, visible brush movement and layered silhouettes.
- **Lotte Reiniger-style silhouette animation**: cut-paper black silhouettes, articulated profile figures, ornate negative space and flat illuminated background. Use the process traits without assuming the user wants a specific copyrighted character.

## Decomposing an absent or invented style

When no reliable entry exists, do not fabricate a canonical definition. Ask or infer the following:

- What is the source medium or object: photograph, painting, print, textile, screen, map, scan or installation?
- What operation makes the marks: exposure, carving, printing, staining, layering, collage, refraction, electrical discharge or digital corruption?
- What are the two or three strongest observable cues in the reference or description?
- Which era, region, maker tradition or technical constraint should be preserved?
- Which traits are essential, and which are optional ageing or decoration?

Compile the answer into the seven-field profile. For a coined hybrid, write the components explicitly in internal reasoning, choose a dominant substrate and process, then translate only the most diagnostic secondary cues. If the term might be a mistaken spelling, use the likely correction only when context makes it clear; otherwise ask.

## Guardrails

- Never claim that a model has learned or can reliably reproduce an obscure style merely because a name appears in training data.
- Never use “vintage”, “arty”, “ethereal”, “tribal”, “scientific” or “experimental” as the entire translation. Pair broad descriptors with process-specific observable cues.
- Do not force historical defects into a clean modern scene when the user asks only for the palette or compositional influence.
- Do not conflate a process, an art movement, a subject genre, an era and a colour grade. State the role of each component internally.
- For culturally specific styles, avoid flattening a living regional tradition into generic decoration. Retain named materials, motifs, script, garment, architecture or technique where supplied.
- Treat source-list difficulty, novelty scores, model capability claims and uncited historical details as non-authoritative. They are not needed for prompt output.

## Compact examples

**Name-only request**: “Mordançage portrait” becomes a black-and-white silver-gelatin portrait with acid-bleached highlights, lifted emulsion veils, branching fissures and cracked relief on paper. The name may remain as an anchor, but the observable signature carries the prompt.

**Hybrid request**: “A cyanotype in the style of a medical scan” becomes a botanical contact-print composition on paper, deep blueprint blue ground, pale plant silhouettes, sectional annotation logic and restrained radiographic layering. Keep the cyanotype substrate dominant and use the scan only for composition and tonal mapping.

**Absent term**: For “rusted neon archive”, infer or ask whether the user wants a photographic print, a sign, or a digital collage. Do not invent a historical movement. Compile the stated substrate, oxidised surface, tube-glow optics, limited palette and archival framing as a new hybrid profile.
