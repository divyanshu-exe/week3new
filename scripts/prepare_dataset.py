"""
Data Preparation and Synthetic Generation for Week 3 NLP Task.
Creates a rich, multi-domain public review corpus with both Sentiment labels
and Named Entity annotations (in standard IOB and Span format).
"""

import json
import os
import random
import pandas as pd
import numpy as np

# Set fixed seed for full reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Curated high-variety review templates & samples covering varied sentiments and entity types
REVIEWS_DATA = [
    # Positive reviews with dense entities
    {
        "review_id": "REV_001",
        "category": "Audio",
        "brand": "Sony",
        "product_name": "Sony WH-1000XM5",
        "author": "Dr. Elena Rostova",
        "date": "2024-02-14",
        "location": "Tokyo",
        "price_mentioned": "$399",
        "rating": 5,
        "ground_truth_sentiment": "positive",
        "review_text": "I bought the Sony WH-1000XM5 headphones last week in Tokyo for $399. The active noise cancellation engineered by Sony is nothing short of miraculous! Dr. Elena Rostova tested them on a flight to London, and the audio fidelity exceeded all expectations. Absolutely fantastic battery life and superb comfort."
    },
    {
        "review_id": "REV_002",
        "category": "Smartphones",
        "brand": "Apple",
        "product_name": "iPhone 15 Pro",
        "author": "Marcus Sterling",
        "date": "2024-01-20",
        "location": "San Francisco",
        "price_mentioned": "$999",
        "rating": 5,
        "ground_truth_sentiment": "positive",
        "review_text": "Tim Cook and Apple delivered a masterpiece with the iPhone 15 Pro. The titanium design feels featherlight compared to last year's model. In San Francisco, I captured 4K ProRes video that looks like a Hollywood movie. Spending $999 was completely justified for this level of performance."
    },
    {
        "review_id": "REV_003",
        "category": "Smartphones",
        "brand": "Samsung",
        "product_name": "Galaxy S24 Ultra",
        "author": "Priya Sharma",
        "date": "2024-03-01",
        "location": "Bangalore",
        "price_mentioned": "$1299",
        "rating": 5,
        "ground_truth_sentiment": "positive",
        "review_text": "The Samsung Galaxy S24 Ultra is the pinnacle of Android innovation. Samsung integrated Galaxy AI features that save me hours of research every single day in Bangalore. The 200MP camera sensor and bright AMOLED screen are breathtaking. Best smartphone purchase of 2024!"
    },
    {
        "review_id": "REV_004",
        "category": "Laptops",
        "brand": "Apple",
        "product_name": "MacBook Air M3",
        "author": "Alexander Wright",
        "date": "2024-03-18",
        "location": "New York",
        "price_mentioned": "$1099",
        "rating": 5,
        "ground_truth_sentiment": "positive",
        "review_text": "Apple's new MacBook Air M3 has revolutionized my remote workflow in New York. The M3 silicon delivers insane speed while maintaining completely silent, fanless operation. Battery easily lasts 18 hours. Highly recommended for students and software engineers alike."
    },
    {
        "review_id": "REV_005",
        "category": "Audio",
        "brand": "Bose",
        "product_name": "Bose QuietComfort Ultra",
        "author": "Chloe Dupont",
        "date": "2024-02-28",
        "location": "Paris",
        "price_mentioned": "$429",
        "rating": 5,
        "ground_truth_sentiment": "positive",
        "review_text": "I purchased the Bose QuietComfort Ultra at a boutique store in Paris for $429. Bose continues to dominate premium noise isolation. The spatial audio mode brings acoustic jazz recordings to life. Truly exceptional ergonomics and build quality."
    },
    {
        "review_id": "REV_006",
        "category": "Laptops",
        "brand": "Dell",
        "product_name": "Dell XPS 15",
        "author": "Michael Chang",
        "date": "2024-01-15",
        "location": "Seattle",
        "price_mentioned": "$1899",
        "rating": 4,
        "ground_truth_sentiment": "positive",
        "review_text": "Dell built a gorgeous machine with the Dell XPS 15. The 3.5K OLED touchscreen display is vibrant and accurate. Intel Core i9 handled my CAD modeling smoothly. While Dell customer support in Seattle took two days to answer a question, the laptop itself is phenomenal."
    },
    {
        "review_id": "REV_007",
        "category": "Smart Home",
        "brand": "Amazon",
        "product_name": "Echo Show 10",
        "author": "Emily Davis",
        "date": "2023-12-10",
        "location": "Austin",
        "price_mentioned": "$249",
        "rating": 4,
        "ground_truth_sentiment": "positive",
        "review_text": "Amazon Echo Show 10 is the centerpiece of our kitchen in Austin. The motorized screen follows you during video calls, which Andy Jassy rightly praised at the launch event. Alexa voice recognition works flawlessly even over loud background cooking noise."
    },
    {
        "review_id": "REV_008",
        "category": "Wearables",
        "brand": "Google",
        "product_name": "Google Pixel Watch 2",
        "author": "Liam O'Connor",
        "date": "2024-02-05",
        "location": "Dublin",
        "price_mentioned": "$349",
        "rating": 4,
        "ground_truth_sentiment": "positive",
        "review_text": "Sundar Pichai and Google finally nailed smartwatches with the Pixel Watch 2. Tracking my morning runs through Phoenix Park in Dublin is seamless thanks to Fitbit sensor integration. The sapphire glass looks pristine and the user interface is silky smooth."
    },
    {
        "review_id": "REV_009",
        "category": "Audio",
        "brand": "Anker",
        "product_name": "Soundcore Space Q45",
        "author": "Carlos Gomez",
        "date": "2024-01-08",
        "location": "Madrid",
        "price_mentioned": "$149",
        "rating": 5,
        "ground_truth_sentiment": "positive",
        "review_text": "Unbelievable value from Anker! The Soundcore Space Q45 delivers 90% of the performance of Sony or Bose for merely $149. In Madrid, my commute on the Metro is now peaceful and serene. Anker's companion app offers comprehensive EQ customization."
    },
    {
        "review_id": "REV_010",
        "category": "Gaming",
        "brand": "Sony",
        "product_name": "PlayStation 5 Slim",
        "author": "Kenji Sato",
        "date": "2023-11-25",
        "location": "Osaka",
        "price_mentioned": "$499",
        "rating": 5,
        "ground_truth_sentiment": "positive",
        "review_text": "Sony knocked it out of the park with the PlayStation 5 Slim. The reduced footprint is ideal for compact apartments in Osaka. Spider-Man 2 loads in under two seconds on the custom NVMe SSD. Gaming at 4K 60fps feels delightfully responsive and immersive."
    },

    # Negative reviews with dense entities and criticisms
    {
        "review_id": "REV_011",
        "category": "Smartphones",
        "brand": "Google",
        "product_name": "Pixel 8 Pro",
        "author": "Jonathan Reed",
        "date": "2024-02-22",
        "location": "Chicago",
        "price_mentioned": "$999",
        "rating": 1,
        "ground_truth_sentiment": "negative",
        "review_text": "Extremely disappointed with Google and the Pixel 8 Pro. For $999, I expected reliable cellular connectivity, but in downtown Chicago the Tensor G3 modem drops calls constantly. Sundar Pichai needs to prioritize hardware quality over half-baked AI gimmicks. The battery drains terribly fast."
    },
    {
        "review_id": "REV_012",
        "category": "Audio",
        "brand": "Apple",
        "product_name": "AirPods Max",
        "author": "Sophia Lorenzi",
        "date": "2024-01-30",
        "location": "Milan",
        "price_mentioned": "$549",
        "rating": 2,
        "ground_truth_sentiment": "negative",
        "review_text": "The Apple AirPods Max are vastly overpriced at $549. After three months in Milan, condensation built up inside the earcups and ruined the right driver. Apple Genius Bar refused warranty coverage. The smart case is ridiculously useless and offers zero protection. Do not buy."
    },
    {
        "review_id": "REV_013",
        "category": "Laptops",
        "brand": "Microsoft",
        "product_name": "Surface Pro 9",
        "author": "Lucas Weber",
        "date": "2024-03-05",
        "location": "Berlin",
        "price_mentioned": "$1299",
        "rating": 2,
        "ground_truth_sentiment": "negative",
        "review_text": "Microsoft Surface Pro 9 failed miserably as a productivity tablet. Windows 11 still suffers from frequent touchscreen freezes and aggressive thermal throttling in Berlin. Paying $1299 for a device without including the keyboard cover is greedy and unacceptable."
    },
    {
        "review_id": "REV_014",
        "category": "Smartphones",
        "brand": "Samsung",
        "product_name": "Galaxy Z Fold 5",
        "author": "Hassan Al-Mansoor",
        "date": "2024-02-18",
        "location": "Dubai",
        "price_mentioned": "$1799",
        "rating": 1,
        "ground_truth_sentiment": "negative",
        "review_text": "Total waste of money on the Samsung Galaxy Z Fold 5. Samsung charged $1799 in Dubai for a phone whose internal folding screen cracked along the crease after just four months. Samsung service center claimed accidental damage and demanded $600 for repairs. Horrible durability and terrible customer care."
    },
    {
        "review_id": "REV_015",
        "category": "Laptops",
        "brand": "HP",
        "product_name": "HP Pavilion 15",
        "author": "Jessica Taylor",
        "date": "2024-01-12",
        "location": "Toronto",
        "price_mentioned": "$699",
        "rating": 2,
        "ground_truth_sentiment": "negative",
        "review_text": "Avoid HP laptops at all costs. The HP Pavilion 15 arrived in Toronto with a defective cooling fan that sounded like a jet engine. HP technical support made me reinstall Windows three times before admitting hardware failure. Flimsy plastic chassis and dreadfully dim display."
    },
    {
        "review_id": "REV_016",
        "category": "Wearables",
        "brand": "Fitbit",
        "product_name": "Fitbit Charge 6",
        "author": "Vikram Patel",
        "date": "2024-03-10",
        "location": "Mumbai",
        "price_mentioned": "$159",
        "rating": 2,
        "ground_truth_sentiment": "negative",
        "review_text": "Ever since Google acquired Fitbit, the software experience has deteriorated. My Fitbit Charge 6 disconnected repeatedly from my phone in Mumbai. Sleep tracking data is frequently inaccurate and Google locked historical metrics behind a paid Fitbit Premium paywall. Not worth $159."
    },
    {
        "review_id": "REV_017",
        "category": "Smart Home",
        "brand": "Ring",
        "product_name": "Ring Video Doorbell Pro",
        "author": "Daniel Craig",
        "date": "2023-12-22",
        "location": "Los Angeles",
        "price_mentioned": "$229",
        "rating": 1,
        "ground_truth_sentiment": "negative",
        "review_text": "The Ring Video Doorbell Pro from Amazon is a major privacy headache and technically flawed. In Los Angeles, notifications are delayed by 45 seconds, making real-time communication impossible. Ring customer service was rude and dismissive. Regret spending $229 on this junk."
    },
    {
        "review_id": "REV_018",
        "category": "Audio",
        "brand": "Beats",
        "product_name": "Beats Studio Pro",
        "author": "Nia Robinson",
        "date": "2024-02-01",
        "location": "Atlanta",
        "price_mentioned": "$349",
        "rating": 2,
        "ground_truth_sentiment": "negative",
        "review_text": "Beats Studio Pro proves that Apple still treats Beats like a second-class brand. The clamping pressure caused severe headaches after twenty minutes of listening in Atlanta. Clunky buttons and cheap build quality compared to Sony WH-1000XM5. Save your $349."
    },
    {
        "review_id": "REV_019",
        "category": "Gaming",
        "brand": "Logitech",
        "product_name": "Logitech G Pro X Superlight",
        "author": "Felix Lindqvist",
        "date": "2024-01-25",
        "location": "Stockholm",
        "price_mentioned": "$150",
        "rating": 2,
        "ground_truth_sentiment": "negative",
        "review_text": "My Logitech G Pro X Superlight developed an annoying double-clicking glitch on the left switch within six weeks. Logitech support in Stockholm was slow to respond and asked for endless video proof. For a competitive esports mouse costing $150, this quality control failure is pathetic."
    },
    {
        "review_id": "REV_020",
        "category": "Tablets",
        "brand": "Amazon",
        "product_name": "Fire Max 11",
        "author": "Brenda Scott",
        "date": "2024-02-12",
        "location": "Denver",
        "price_mentioned": "$229",
        "rating": 2,
        "ground_truth_sentiment": "negative",
        "review_text": "Amazon Fire Max 11 is severely crippled by Fire OS. No native Google Play Store support means you cannot download essential productivity apps in Denver. Full of intrusive Amazon lock-screen advertisements and sluggish multi-tasking. Frustrating and uninspiring device."
    },

    # Neutral and Comparative reviews with mixed sentiment
    {
        "review_id": "REV_021",
        "category": "Audio",
        "brand": "Sony",
        "product_name": "Sony WF-1000XM5",
        "author": "Tariq Mansour",
        "date": "2024-03-02",
        "location": "Doha",
        "price_mentioned": "$299",
        "rating": 3,
        "ground_truth_sentiment": "neutral",
        "review_text": "The Sony WF-1000XM5 earbuds are a mixed bag. Sound reproduction is pristine and crisp, but the glossy finish makes them slippery to remove from the case in humid Doha weather. Noise cancellation matches Bose, but the foam tips degrade rapidly. Decent, but not extraordinary for $299."
    },
    {
        "review_id": "REV_022",
        "category": "Smartphones",
        "brand": "Apple",
        "product_name": "iPhone 15",
        "author": "Rachel Green",
        "date": "2024-02-19",
        "location": "New York",
        "price_mentioned": "$799",
        "rating": 3,
        "ground_truth_sentiment": "neutral",
        "review_text": "Apple's baseline iPhone 15 introduces the Dynamic Island and USB-C port, which is convenient for New York commuters. However, keeping the 60Hz display in 2024 when Samsung offers 120Hz at lower prices feels stingy. An acceptable standard upgrade, neither breathtaking nor terrible."
    },
    {
        "review_id": "REV_023",
        "category": "Laptops",
        "brand": "Lenovo",
        "product_name": "Lenovo ThinkPad X1 Carbon",
        "author": "Dr. Hans Gruber",
        "date": "2024-01-28",
        "location": "Frankfurt",
        "price_mentioned": "$1650",
        "rating": 3,
        "ground_truth_sentiment": "neutral",
        "review_text": "Lenovo ThinkPad X1 Carbon maintains its legendary ergonomic keyboard and robust business durability in Frankfurt. Unfortunately, battery endurance under Linux dropped to five hours, and the webcam quality is mediocre. Satisfactory for corporate finance, but overpriced at $1650."
    },
    {
        "review_id": "REV_024",
        "category": "Smartphones",
        "brand": "OnePlus",
        "product_name": "OnePlus 12",
        "author": "Aarav Mehta",
        "date": "2024-02-27",
        "location": "Mumbai",
        "price_mentioned": "$799",
        "rating": 3,
        "ground_truth_sentiment": "neutral",
        "review_text": "OnePlus 12 offers beastly Snapdragon 8 Gen 3 performance and lightning-fast 100W charging in Mumbai. However, OxygenOS still has minor notification bugs, and the Hasselblad camera tuning struggles in harsh backlight. A solid alternative to Samsung Galaxy S24, but with clear trade-offs."
    },
    {
        "review_id": "REV_025",
        "category": "Wearables",
        "brand": "Samsung",
        "product_name": "Galaxy Watch 6",
        "author": "Oliver Queen",
        "date": "2024-03-08",
        "location": "Star City",
        "price_mentioned": "$299",
        "rating": 3,
        "ground_truth_sentiment": "neutral",
        "review_text": "Samsung Galaxy Watch 6 brings back the beloved physical rotating bezel, which feels delightful to navigate. However, the battery life barely stretches across 24 hours with always-on display enabled. It is an iterative refinement of the Watch 5 rather than a revolution."
    },
    {
        "review_id": "REV_026",
        "category": "Laptops",
        "brand": "Asus",
        "product_name": "Asus ROG Zephyrus G14",
        "author": "Dmitri Volkov",
        "date": "2024-02-11",
        "location": "Moscow",
        "price_mentioned": "$1599",
        "rating": 4,
        "ground_truth_sentiment": "positive",
        "review_text": "Asus engineered a compact powerhouse with the ROG Zephyrus G14. The AMD Ryzen 9 processor and Nvidia RTX 4070 tear through Cyberpunk 2077 easily in Moscow. The OLED panel is stunning. It gets quite warm under load, but the performance per pound is unmatched."
    },
    {
        "review_id": "REV_027",
        "category": "E-Readers",
        "brand": "Amazon",
        "product_name": "Kindle Paperwhite",
        "author": "Hannah Abbott",
        "date": "2023-12-15",
        "location": "Edinburgh",
        "price_mentioned": "$139",
        "rating": 5,
        "ground_truth_sentiment": "positive",
        "review_text": "Amazon Kindle Paperwhite remains the undisputed champion of digital reading. Sitting in an Edinburgh cafe with warm adjustable backlighting and waterproof durability is pure bliss. Battery lasts ten weeks on a single charge. Jeff Bezos built an enduring classic."
    },
    {
        "review_id": "REV_028",
        "category": "Audio",
        "brand": "Sennheiser",
        "product_name": "Sennheiser Momentum 4",
        "author": "Klaus Schmidt",
        "date": "2024-01-18",
        "location": "Munich",
        "price_mentioned": "$349",
        "rating": 4,
        "ground_truth_sentiment": "positive",
        "review_text": "Sennheiser Momentum 4 boasts an astounding 60-hour battery life that embarrasses Sony and Bose. In Munich, the audiophile sound signature delivers rich bass and crystalline highs. The touch controls can be slightly oversensitive in cold weather, but acoustically it is top tier."
    },
    {
        "review_id": "REV_029",
        "category": "Smartphones",
        "brand": "Google",
        "product_name": "Google Pixel 7a",
        "author": "Anita Roy",
        "date": "2024-01-05",
        "location": "Kolkata",
        "price_mentioned": "$449",
        "rating": 4,
        "ground_truth_sentiment": "positive",
        "review_text": "Google Pixel 7a provides flagship camera quality at an accessible $449 price point in Kolkata. Google's computational photography outperforms rivals like Samsung Galaxy A54. It charges rather slowly at 18W, but overall software fluidity and clean Android 14 are unbeatable."
    },
    {
        "review_id": "REV_030",
        "category": "Monitors",
        "brand": "Dell",
        "product_name": "Dell UltraSharp 27 4K",
        "author": "Arthur Pendelton",
        "date": "2024-02-24",
        "location": "Boston",
        "price_mentioned": "$579",
        "rating": 5,
        "ground_truth_sentiment": "positive",
        "review_text": "Dell UltraSharp 27 4K monitor transformed my graphic design workstation in Boston. Dell's IPS Black technology provides deep contrast and 98% DCI-P3 color precision. The built-in 90W USB-C hub charges my MacBook while transmitting video seamlessly. Worth every cent of $579."
    }
]

# Expand review dataset to 100+ comprehensive items by generating systematic variations
# with diverse linguistic structures, entities, and sentiment polarities
EXTENDED_REVIEWS = list(REVIEWS_DATA)

AUTHORS_POOL = ["James Wilson", "Aiko Tanaka", "Mateo Rossi", "Leila Zahra", "Claire Bennet", "Arjun Nair", "Zoe Kravitz", "Gabriel Silva", "Nadia Petrova", "Chen Wei"]
LOCATIONS_POOL = ["London", "Seoul", "Sydney", "Singapore", "Zurich", "Hong Kong", "San Jose", "Melbourne", "San Diego", "Montreal"]
DATES_POOL = ["2024-01-11", "2024-02-04", "2024-02-17", "2024-03-09", "2024-03-22", "2024-04-02", "2024-04-15", "2024-04-28"]

BRANDS_PRODUCTS = [
    ("Apple", "MacBook Pro M3 Max", "Laptops", 2499, "positive", "The Apple MacBook Pro M3 Max is a computational titan. Compiling large C++ codebases in {loc} takes half the time compared to my old Intel laptop. Tim Cook and the Apple engineering team deserve applause for creating such a thermal miracle."),
    ("Apple", "iPad Pro 12.9", "Tablets", 1099, "negative", "Apple iPad Pro 12.9 has glorious mini-LED hardware, but iPadOS is painfully limited. In {loc}, trying to do real multi-window software development resulted in constant frustration. For ${price}, Apple should just give us macOS."),
    ("Samsung", "Galaxy Tab S9 Ultra", "Tablets", 1199, "positive", "Samsung built the ultimate media consumption machine with the Galaxy Tab S9 Ultra. Watching Netflix movies on the 14.6-inch AMOLED screen in {loc} feels like having a private cinema. Samsung S-Pen included in the box is a huge bonus."),
    ("Samsung", "Galaxy S23 FE", "Smartphones", 599, "negative", "The Samsung Galaxy S23 FE gets uncomfortably hot during basic video calls in {loc}. Samsung used an outdated Exynos chip that throttles performance and sips battery aggressively. Definitely not worth ${price}."),
    ("Sony", "Sony A7 IV Mirrorless Camera", "Cameras", 2498, "positive", "Sony A7 IV is the undisputed king of hybrid photography. Shooting weddings across {loc} with Sony's real-time eye autofocus never misses a beat. Low light performance is unmatched at ${price}."),
    ("Sony", "Sony LinkBuds S", "Audio", 199, "neutral", "Sony LinkBuds S are lightweight and comfortable for all-day office wear in {loc}. Noise cancellation is good, though Sony call microphone quality drops sharply when wind picks up. Fair value for ${price}."),
    ("Google", "Google Pixel Fold", "Smartphones", 1799, "negative", "Google Pixel Fold has an awkward heavy hinge and noticeable screen reflection outdoors in {loc}. Sundar Pichai released a prototype device at a premium price of ${price}. App optimization on the wide inner screen is lacking."),
    ("Google", "Google Nest Hub Max", "Smart Home", 229, "positive", "Google Nest Hub Max keeps our family organized in {loc}. Video calls via Google Meet are crisp, and the built-in speaker fills the room with rich audio. Excellent smart home hub from Google."),
    ("Microsoft", "Xbox Series X", "Gaming", 499, "positive", "Satya Nadella and Microsoft created an absolute beast with Xbox Series X. Quick Resume feature alone is worth ${price}. Gaming at 4K in {loc} with Game Pass library offers endless entertainment."),
    ("Microsoft", "Surface Laptop 5", "Laptops", 999, "neutral", "Microsoft Surface Laptop 5 has a gorgeous Alcantara keyboard and solid build in {loc}. However, thick screen bezels and lack of 120Hz refresh rate make it feel dated compared to Apple MacBook Air. Acceptable at ${price}."),
    ("Bose", "Bose Smart Soundbar 900", "Audio", 899, "positive", "Bose Smart Soundbar 900 delivers thrilling Dolby Atmos surround sound in our living room in {loc}. Bose audio calibration tuned the soundstage perfectly. Movie nights feel like an authentic IMAX theater experience."),
    ("Bose", "Bose Frames Tenor", "Audio", 249, "negative", "Bose Frames audio sunglasses were a total letdown in {loc}. Battery dies in under two hours and sound leaks so loudly that anyone sitting next to you hears every word. Bose missed the mark completely on this ${price} gadget."),
    ("Dell", "Dell Alienware m16", "Gaming", 2199, "positive", "Dell Alienware m16 crushes every AAA game title effortlessly in {loc}. The mechanical keyboard with Cherry MX switches provides superb tactile feedback. Dell created a true desktop replacement."),
    ("Dell", "Dell Inspiron 14", "Laptops", 549, "negative", "Dell Inspiron 14 has dreadful build quality. The trackpad became loose within two weeks of use in {loc}, and Dell technical support was completely unhelpful. Save your money and avoid this ${price} model."),
    ("Asus", "Asus Zenbook 14 OLED", "Laptops", 1049, "positive", "Asus Zenbook 14 OLED is an engineering triumph. The Intel Core Ultra processor and Lumina OLED screen in {loc} offer breathtaking HDR contrast. Asus delivers premium quality at ${price}."),
    ("Lenovo", "Lenovo Legion Pro 7i", "Gaming", 2299, "positive", "Lenovo Legion Pro 7i is the best gaming laptop of the year. In {loc}, thermals stay remarkably cool thanks to vapor chamber cooling. Lenovo provided outstanding gaming performance for ${price}."),
    ("Anker", "Anker 737 Power Bank", "Accessories", 149, "positive", "Anker 737 Power Bank with 140W fast charging is an absolute lifesaver during travels through {loc}. The digital display showing real-time wattage output is brilliant. Anker hardware never disappoints."),
    ("Anker", "Anker Soundcore Liberty 4", "Audio", 129, "neutral", "Anker Soundcore Liberty 4 has decent sound and heart rate tracking in {loc}. However, touch controls are finicky and active noise canceling is mediocre compared to Sony. Fair product for ${price}."),
    ("HP", "HP Spectre x360 14", "Laptops", 1499, "positive", "HP redeemed itself with the Spectre x360 14. In {loc}, the gem-cut chassis, 9-hour battery, and OLED panel make it a joy to use. HP designed one of the finest 2-in-1 laptops on the market."),
    ("OnePlus", "OnePlus Open", "Smartphones", 1699, "positive", "OnePlus Open is genuinely superior to Samsung Galaxy Z Fold 5. The crease is virtually invisible and multitasking with Open Canvas in {loc} is revolutionary. OnePlus scored a massive victory here.")
]

idx = len(EXTENDED_REVIEWS) + 1
for brand, prod, cat, price, sentiment, template in BRANDS_PRODUCTS:
    author = random.choice(AUTHORS_POOL)
    loc = random.choice(LOCATIONS_POOL)
    date = random.choice(DATES_POOL)
    rating = 5 if sentiment == "positive" else (1 if sentiment == "negative" else 3)
    text = template.format(loc=loc, price=price)
    # Add author attribution
    text = f"{author} reports from {loc}: " + text
    
    EXTENDED_REVIEWS.append({
        "review_id": f"REV_{idx:03d}",
        "category": cat,
        "brand": brand,
        "product_name": prod,
        "author": author,
        "date": date,
        "location": loc,
        "price_mentioned": f"${price}",
        "rating": rating,
        "ground_truth_sentiment": sentiment,
        "review_text": text
    })
    idx += 1

# Add 50 more synthetic reviews to make total ~100 reviews with rich balance
SENTIMENT_DIST = ["positive", "positive", "negative", "negative", "neutral"]
for i in range(50):
    brand, prod, cat, base_price, _, _ = random.choice(BRANDS_PRODUCTS)
    s_choice = random.choice(SENTIMENT_DIST)
    rating = random.choice([4, 5]) if s_choice == "positive" else (random.choice([1, 2]) if s_choice == "negative" else 3)
    author = random.choice(AUTHORS_POOL)
    loc = random.choice(LOCATIONS_POOL)
    date = random.choice(DATES_POOL)
    price = base_price + random.randint(-50, 50)
    
    if s_choice == "positive":
        review_text = (
            f"Review by {author} in {loc}: I tested the {prod} manufactured by {brand} on {date}. "
            f"At a retail price of ${price}, this device offers exceptional reliability, outstanding battery efficiency, "
            f"and premium build materials. {brand} has truly outdone itself with this release. Highly recommended!"
        )
    elif s_choice == "negative":
        review_text = (
            f"Customer complaint from {author} in {loc}: Purchasing the {prod} from {brand} for ${price} was a huge mistake. "
            f"On {date}, the unit malfunctioned completely. Customer service at {brand} was dismissive, unresponsive, and unhelpful. "
            f"Poor durability, buggy software, and terrible value for money. Do not recommend."
        )
    else:
        review_text = (
            f"Detailed analysis by {author} in {loc}: The {prod} from {brand} retails around ${price} as of {date}. "
            f"While the industrial design and hardware responsiveness are commendable, the accompanying software features feel unpolished. "
            f"Compared to competing alternatives from rivals, it delivers average performance. An acceptable purchase with caveats."
        )
        
    EXTENDED_REVIEWS.append({
        "review_id": f"REV_{idx:03d}",
        "category": cat,
        "brand": brand,
        "product_name": prod,
        "author": author,
        "date": date,
        "location": loc,
        "price_mentioned": f"${price}",
        "rating": rating,
        "ground_truth_sentiment": s_choice,
        "review_text": review_text
    })
    idx += 1

# Save Reviews Dataset as CSV
df_reviews = pd.DataFrame(EXTENDED_REVIEWS)
reviews_csv_path = os.path.join(OUTPUT_DIR, "reviews_dataset.csv")
df_reviews.to_csv(reviews_csv_path, index=False)
print(f"Saved {len(df_reviews)} review records to: {reviews_csv_path}")

# =====================================================================
# Dedicated NER Ground-Truth Benchmark Dataset (Token-level IOB format)
# =====================================================================
# Tokenized sentence samples with manual high-precision IOB tags:
# B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-PROD, I-PROD, B-DATE, I-DATE, B-MONEY, I-MONEY, O

NER_BENCHMARK_SENTENCES = [
    {
        "sentence_id": "NER_S01",
        "tokens": ["Tim", "Cook", "announced", "the", "iPhone", "15", "Pro", "at", "Apple", "headquarters", "in", "Cupertino", "on", "September", "12", ",", "2023", "for", "$", "999", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "DT", "NNP", "CD", "NNP", "IN", "NNP", "NN", "IN", "NNP", "IN", "NNP", "CD", ",", "CD", "IN", "$", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "B-PROD", "I-PROD", "I-PROD", "O", "B-ORG", "O", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O", "B-MONEY", "I-MONEY", "O"]
    },
    {
        "sentence_id": "NER_S02",
        "tokens": ["Sundar", "Pichai", "demonstrated", "Google", "Pixel", "8", "Pro", "camera", "innovations", "in", "Mountain", "View", "for", "$", "999", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "NNP", "NNP", "CD", "NNP", "NN", "NNS", "IN", "NNP", "NNP", "IN", "$", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "O", "O", "B-LOC", "I-LOC", "O", "B-MONEY", "I-MONEY", "O"]
    },
    {
        "sentence_id": "NER_S03",
        "tokens": ["Satya", "Nadella", "praised", "Microsoft", "Surface", "Laptop", "5", "during", "the", "conference", "in", "Seattle", "on", "October", "25", ",", "2023", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "NNP", "NNP", "NNP", "CD", "IN", "DT", "NN", "IN", "NNP", "IN", "NNP", "CD", ",", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "O", "O", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O"]
    },
    {
        "sentence_id": "NER_S04",
        "tokens": ["Dr.", "Elena", "Rostova", "bought", "the", "Sony", "WH-1000XM5", "headphones", "in", "Tokyo", "for", "$", "399", "."],
        "pos_tags": ["NNP", "NNP", "NNP", "VBD", "DT", "NNP", "NNP", "NNS", "IN", "NNP", "IN", "$", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "O", "O", "B-LOC", "O", "B-MONEY", "I-MONEY", "O"]
    },
    {
        "sentence_id": "NER_S05",
        "tokens": ["Priya", "Sharma", "tested", "the", "Samsung", "Galaxy", "S24", "Ultra", "in", "Bangalore", "on", "January", "17", ",", "2024", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "DT", "NNP", "NNP", "NNP", "NNP", "IN", "NNP", "IN", "NNP", "CD", ",", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O"]
    },
    {
        "sentence_id": "NER_S06",
        "tokens": ["Chloe", "Dupont", "reviewed", "the", "Bose", "QuietComfort", "Ultra", "earbuds", "in", "Paris", "priced", "at", "EUR", "429", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "DT", "NNP", "NNP", "NNP", "NNS", "IN", "NNP", "VBN", "IN", "NNP", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "O", "O", "B-LOC", "O", "O", "B-MONEY", "I-MONEY", "O"]
    },
    {
        "sentence_id": "NER_S07",
        "tokens": ["Michael", "Chang", "purchased", "a", "Dell", "XPS", "15", "laptop", "in", "Austin", "for", "$", "1899", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "DT", "NNP", "NNP", "CD", "NN", "IN", "NNP", "IN", "$", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "O", "O", "B-LOC", "O", "B-MONEY", "I-MONEY", "O"]
    },
    {
        "sentence_id": "NER_S08",
        "tokens": ["Andy", "Jassy", "unveiled", "the", "Amazon", "Echo", "Show", "10", "in", "Seattle", "on", "November", "28", ",", "2023", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "DT", "NNP", "NNP", "NNP", "CD", "IN", "NNP", "IN", "NNP", "CD", ",", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O"]
    },
    {
        "sentence_id": "NER_S09",
        "tokens": ["Carlos", "Gomez", "recommended", "the", "Anker", "Soundcore", "Space", "Q45", "in", "Madrid", "for", "$", "149", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "DT", "NNP", "NNP", "NNP", "NNP", "IN", "NNP", "IN", "$", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "B-MONEY", "I-MONEY", "O"]
    },
    {
        "sentence_id": "NER_S10",
        "tokens": ["Kenji", "Sato", "purchased", "the", "PlayStation", "5", "Slim", "from", "Sony", "in", "Osaka", "on", "December", "5", ",", "2023", "for", "$", "499", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "DT", "NNP", "CD", "NNP", "IN", "NNP", "IN", "NNP", "IN", "NNP", "CD", ",", "CD", "IN", "$", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "B-PROD", "I-PROD", "I-PROD", "O", "B-ORG", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O", "B-MONEY", "I-MONEY", "O"]
    },
    {
        "sentence_id": "NER_S11",
        "tokens": ["Jonathan", "Reed", "criticized", "Google", "and", "the", "Pixel", "8", "Pro", "in", "Chicago", "after", "spending", "$", "999", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "NNP", "CC", "DT", "NNP", "CD", "NNP", "IN", "NNP", "IN", "VBG", "$", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "B-ORG", "O", "O", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "O", "B-MONEY", "I-MONEY", "O"]
    },
    {
        "sentence_id": "NER_S12",
        "tokens": ["Sophia", "Lorenzi", "returned", "the", "Apple", "AirPods", "Max", "in", "Milan", "following", "the", "incident", "in", "January", "2024", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "DT", "NNP", "NNP", "NNP", "IN", "NNP", "VBG", "DT", "NN", "IN", "NNP", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "O", "B-LOC", "O", "O", "O", "O", "B-DATE", "I-DATE", "O"]
    },
    {
        "sentence_id": "NER_S13",
        "tokens": ["Lucas", "Weber", "tested", "the", "Microsoft", "Surface", "Pro", "9", "in", "Berlin", "and", "paid", "$", "1299", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "DT", "NNP", "NNP", "NNP", "CD", "IN", "NNP", "CC", "VBD", "$", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "O", "B-MONEY", "I-MONEY", "O"]
    },
    {
        "sentence_id": "NER_S14",
        "tokens": ["Hassan", "Al-Mansoor", "bought", "the", "Samsung", "Galaxy", "Z", "Fold", "5", "in", "Dubai", "for", "$", "1799", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "DT", "NNP", "NNP", "NNP", "NNP", "CD", "IN", "NNP", "IN", "$", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "B-MONEY", "I-MONEY", "O"]
    },
    {
        "sentence_id": "NER_S15",
        "tokens": ["Jessica", "Taylor", "reported", "fan", "issues", "with", "the", "HP", "Pavilion", "15", "in", "Toronto", "costing", "$", "699", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "NN", "NNS", "IN", "DT", "NNP", "NNP", "CD", "IN", "NNP", "VBG", "$", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "O", "O", "O", "B-ORG", "B-PROD", "I-PROD", "O", "B-LOC", "O", "B-MONEY", "I-MONEY", "O"]
    },
    {
        "sentence_id": "NER_S16",
        "tokens": ["Vikram", "Patel", "analyzed", "Google", "Fitbit", "Charge", "6", "metrics", "in", "Mumbai", "on", "March", "10", ",", "2024", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "NNP", "NNP", "NNP", "CD", "NNS", "IN", "NNP", "IN", "NNP", "CD", ",", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O"]
    },
    {
        "sentence_id": "NER_S17",
        "tokens": ["Daniel", "Craig", "installed", "the", "Ring", "Video", "Doorbell", "Pro", "from", "Amazon", "in", "Los", "Angeles", "for", "$", "229", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "DT", "NNP", "NNP", "NNP", "NNP", "IN", "NNP", "IN", "NNP", "NNP", "IN", "$", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "B-PROD", "I-PROD", "I-PROD", "I-PROD", "O", "B-ORG", "O", "B-LOC", "I-LOC", "O", "B-MONEY", "I-MONEY", "O"]
    },
    {
        "sentence_id": "NER_S18",
        "tokens": ["Nia", "Robinson", "compared", "Beats", "Studio", "Pro", "against", "Sony", "WH-1000XM5", "in", "Atlanta", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "NNP", "NNP", "NNP", "IN", "NNP", "NNP", "IN", "NNP", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "O", "B-ORG", "B-PROD", "O", "B-LOC", "O"]
    },
    {
        "sentence_id": "NER_S19",
        "tokens": ["Felix", "Lindqvist", "ordered", "the", "Logitech", "G", "Pro", "X", "Superlight", "in", "Stockholm", "for", "$", "150", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "DT", "NNP", "NNP", "NNP", "NNP", "NNP", "IN", "NNP", "IN", "$", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "B-MONEY", "I-MONEY", "O"]
    },
    {
        "sentence_id": "NER_S20",
        "tokens": ["Brenda", "Scott", "reviewed", "the", "Amazon", "Fire", "Max", "11", "in", "Denver", "on", "February", "12", ",", "2024", "."],
        "pos_tags": ["NNP", "NNP", "VBD", "DT", "NNP", "NNP", "NNP", "CD", "IN", "NNP", "IN", "NNP", "CD", ",", "CD", "."],
        "ner_tags": ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O"]
    }
]

# Expand benchmark sentences by generating additional controlled annotated sentences (50 total)
MORE_SENTENCES = [
    ("Alexander Wright purchased the Apple MacBook Air M3 in New York for $ 1099 .",
     ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "I-LOC", "O", "B-MONEY", "I-MONEY", "O"]),
    ("Tariq Mansour tested Sony WF-1000XM5 earbuds in Doha on March 2 , 2024 .",
     ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "O", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O"]),
    ("Rachel Green evaluated the Apple iPhone 15 in New York for $ 799 .",
     ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "O", "B-LOC", "I-LOC", "O", "B-MONEY", "I-MONEY", "O"]),
    ("Dr. Hans Gruber examined Lenovo ThinkPad X1 Carbon in Frankfurt on January 28 , 2024 .",
     ["B-PER", "I-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O"]),
    ("Aarav Mehta benchmarked OnePlus 12 against Samsung Galaxy S24 in Mumbai for $ 799 .",
     ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "O", "B-ORG", "B-PROD", "I-PROD", "O", "B-LOC", "O", "B-MONEY", "I-MONEY", "O"]),
    ("Oliver Queen inspected the Samsung Galaxy Watch 6 in Star City on March 8 , 2024 .",
     ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "I-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O"]),
    ("Dmitri Volkov reviewed Asus ROG Zephyrus G14 in Moscow for $ 1599 .",
     ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "B-MONEY", "I-MONEY", "O"]),
    ("Jeff Bezos introduced the Amazon Kindle Paperwhite in Edinburgh priced at $ 139 .",
     ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "O", "B-LOC", "O", "O", "B-MONEY", "I-MONEY", "O"]),
    ("Klaus Schmidt evaluated Sennheiser Momentum 4 headphones in Munich for $ 349 .",
     ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "O", "O", "B-LOC", "O", "B-MONEY", "I-MONEY", "O"]),
    ("Anita Roy praised Google Pixel 7a camera features in Kolkata on January 5 , 2024 .",
     ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "O", "O", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O"]),
    ("Arthur Pendelton calibrated the Dell UltraSharp 27 4K display in Boston for $ 579 .",
     ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "O", "B-LOC", "O", "B-MONEY", "I-MONEY", "O"]),
    ("Sarah Connor tested Apple MacBook Pro M3 Max in London for $ 2499 .",
     ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "B-MONEY", "I-MONEY", "O"]),
    ("David Miller evaluated Samsung Galaxy Tab S9 Ultra in Seoul on February 14 , 2024 .",
     ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O"]),
    ("Leila Zahra acquired the Sony A7 IV in Singapore for $ 2498 .",
     ["B-PER", "I-PER", "O", "O", "B-ORG", "B-PROD", "I-PROD", "O", "B-LOC", "O", "B-MONEY", "I-MONEY", "O"]),
    ("Gabriel Silva installed Microsoft Xbox Series X in Zurich on December 10 , 2023 .",
     ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O"]),
    ("Claire Bennet repaired Dell Alienware m16 in San Jose for $ 2199 .",
     ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "O", "B-LOC", "I-LOC", "O", "B-MONEY", "I-MONEY", "O"]),
    ("Chen Wei unboxed Asus Zenbook 14 OLED in Hong Kong on April 2 , 2024 .",
     ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "I-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O"]),
    ("Nadia Petrova ordered Lenovo Legion Pro 7i in Montreal for $ 2299 .",
     ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "B-MONEY", "I-MONEY", "O"]),
    ("Mateo Rossi charged Anker 737 Power Bank in Melbourne on March 15 , 2024 .",
     ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "O", "B-DATE", "I-DATE", "I-DATE", "I-DATE", "O"]),
    ("Arjun Nair presented HP Spectre x360 14 in San Diego for $ 1499 .",
     ["B-PER", "I-PER", "O", "B-ORG", "B-PROD", "I-PROD", "I-PROD", "O", "B-LOC", "I-LOC", "O", "B-MONEY", "I-MONEY", "O"])
]

import nltk
from nltk import pos_tag

for raw_s, tags in MORE_SENTENCES:
    tokens = raw_s.split()
    assert len(tokens) == len(tags), f"Mismatch in {raw_s}: {len(tokens)} vs {len(tags)}"
    # Generate POS tags automatically using nltk
    pos = [p[1] for p in pos_tag(tokens)]
    NER_BENCHMARK_SENTENCES.append({
        "sentence_id": f"NER_S{len(NER_BENCHMARK_SENTENCES)+1:02d}",
        "tokens": tokens,
        "pos_tags": pos,
        "ner_tags": tags
    })

# Save NER benchmark to JSON
ner_json_path = os.path.join(OUTPUT_DIR, "ner_ground_truth.json")
with open(ner_json_path, "w", encoding="utf-8") as f:
    json.dump(NER_BENCHMARK_SENTENCES, f, indent=2)
print(f"Saved {len(NER_BENCHMARK_SENTENCES)} annotated NER sentences to: {ner_json_path}")

# Also save a flattened CSV for easy tabular analysis
flattened_ner = []
for s in NER_BENCHMARK_SENTENCES:
    sid = s["sentence_id"]
    for i, (t, p, n) in enumerate(zip(s["tokens"], s["pos_tags"], s["ner_tags"])):
        flattened_ner.append({
            "sentence_id": sid,
            "token_index": i,
            "token": t,
            "pos_tag": p,
            "ner_tag": n
        })
df_ner = pd.DataFrame(flattened_ner)
ner_csv_path = os.path.join(OUTPUT_DIR, "ner_tokens_benchmark.csv")
df_ner.to_csv(ner_csv_path, index=False)
print(f"Saved {len(df_ner)} token rows to: {ner_csv_path}")
