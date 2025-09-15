# File: config.py
"""
Contains the static course catalog for the Streamlit UI.
This defines the relationship between courses, fields, and subjects.
"""
COURSE_CATALOG = {
    "B.Tech": {
        "CSE": ["Data Structures & Algorithms", "Operating Systems", "DBMS", "Computer Networks", "Software Engineering", "Machine Learning", "Web Development", "Artificial Intelligence", "Cybersecurity", "Compiler Design"],
        "Mechanical": ["Thermodynamics", "Fluid Mechanics", "Heat Transfer", "Manufacturing Technology", "Machine Design", "Strength of Materials", "Automobile Engineering", "Production Engineering", "CAD/CAM", "Industrial Engineering"],
        "ECE": ["Digital Electronics", "Analog Electronics", "Communication Systems", "Signal Processing", "VLSI Design", "Microprocessors", "Control Systems", "Electromagnetic Theory", "RF Engineering", "Embedded Systems"],
        "EEE": ["Power Systems", "Electrical Machines", "Power Electronics", "Control Systems", "High Voltage Engineering", "Renewable Energy", "Protection & Switchgear", "Electric Drives", "Grid Integration", "Power Quality"],
        "Civil": ["Structural Engineering", "Geotechnical Engineering", "Transportation Engineering", "Environmental Engineering", "Hydraulics", "Construction Management", "Concrete Technology", "Surveying", "Urban Planning", "Water Resources"],
        "Chemical": ["Chemical Reaction Engineering", "Mass Transfer", "Heat Transfer", "Process Control", "Thermodynamics", "Fluid Mechanics", "Separation Processes", "Process Design", "Petrochemicals", "Environmental Engineering"],
        "IT": ["Programming Languages", "Software Testing", "System Analysis", "Network Security", "Mobile App Development", "Cloud Computing", "Data Analytics", "DevOps", "UI/UX Design", "Project Management"],
        "Biotechnology": ["Molecular Biology", "Genetic Engineering", "Bioprocess Engineering", "Biochemistry", "Microbiology", "Cell Biology", "Bioinformatics", "Environmental Biotechnology", "Medical Biotechnology", "Industrial Biotechnology"],
        "Aerospace": ["Aerodynamics", "Aircraft Structures", "Propulsion", "Flight Mechanics", "Avionics", "Space Technology", "Control Systems", "Materials Science", "Computational Fluid Dynamics", "Satellite Technology"],
        "Automobile": ["IC Engines", "Vehicle Dynamics", "Automotive Electronics", "Transmission Systems", "Chassis Design", "Alternative Fuels", "Vehicle Safety", "Hybrid Vehicles", "Manufacturing Processes", "Quality Control"]
    },
    
    "B.Com": {
        "General": ["Financial Accounting", "Business Law", "Macroeconomics", "Microeconomics", "Business Statistics", "Corporate Accounting", "Income Tax", "Auditing", "Cost Accounting", "Business Communication"],
        "Honours": ["Advanced Accounting", "Financial Management", "Investment Analysis", "International Business", "Strategic Management", "Research Methodology", "Banking & Insurance", "Capital Markets", "Financial Derivatives", "Portfolio Management"],
        "Computer Applications": ["Computer Fundamentals", "Programming in C", "Database Management", "E-Commerce", "Digital Marketing", "MS Office", "Accounting Software", "Web Design", "Data Analysis", "Business Intelligence"]
    },
    
    "BBA": {
        "General": ["Principles of Management", "Marketing Management", "Financial Management", "Human Resource Management", "Business Economics", "Organizational Behavior", "Operations Management", "Strategic Management", "Business Ethics", "Entrepreneurship"],
        "Marketing": ["Consumer Behavior", "Digital Marketing", "Brand Management", "Sales Management", "Market Research", "Advertising", "Retail Management", "International Marketing", "Services Marketing", "Marketing Analytics"],
        "Finance": ["Corporate Finance", "Investment Banking", "Risk Management", "Financial Markets", "Portfolio Management", "Derivatives", "International Finance", "Financial Planning", "Mergers & Acquisitions", "Behavioral Finance"],
        "HR": ["Recruitment & Selection", "Training & Development", "Performance Management", "Compensation Management", "Labor Laws", "Industrial Relations", "Leadership", "Change Management", "HR Analytics", "Talent Management"],
        "International Business": ["Global Business Environment", "Cross Cultural Management", "International Trade", "Export-Import Procedures", "Foreign Exchange", "Global Supply Chain", "International Marketing", "Multinational Corporations", "Trade Policies", "Global Economics"]
    },
    
    "BA LLB": {
        "Law": ["Constitutional Law", "Criminal Law", "Contract Law", "Tort Law", "Property Law", "Family Law", "Administrative Law", "Corporate Law", "Environmental Law", "Intellectual Property Rights"],
        "Criminal Law": ["Indian Penal Code", "Criminal Procedure Code", "Evidence Law", "Forensic Science", "Criminology", "Victimology", "Juvenile Justice", "Cyber Crimes", "White Collar Crimes", "Prison Reforms"],
        "Corporate Law": ["Company Law", "Securities Law", "Competition Law", "Banking Law", "Insurance Law", "Tax Law", "Labor Law", "International Trade Law", "Mergers & Acquisitions", "Corporate Governance"],
        "Constitutional Law": ["Fundamental Rights", "Directive Principles", "Amendment Procedures", "Federalism", "Emergency Provisions", "Election Law", "Parliamentary Procedures", "Judicial Review", "Constitutional History", "Comparative Constitution"]
    },
    
    "BA": {
        "English": ["British Literature", "American Literature", "Indian English Literature", "Poetry", "Drama", "Fiction", "Literary Criticism", "Linguistics", "Creative Writing", "Comparative Literature"],
        "History": ["Ancient Indian History", "Medieval Indian History", "Modern Indian History", "World History", "Art History", "Social History", "Economic History", "Political History", "Historiography", "Archaeological Studies"],
        "Political Science": ["Indian Government & Politics", "Comparative Politics", "International Relations", "Political Theory", "Public Administration", "Political Economy", "Foreign Policy", "Democracy Studies", "Political Sociology", "Strategic Studies"],
        "Economics": ["Microeconomics", "Macroeconomics", "Indian Economy", "Development Economics", "International Economics", "Monetary Economics", "Public Finance", "Econometrics", "Environmental Economics", "Behavioral Economics"],
        "Psychology": ["General Psychology", "Developmental Psychology", "Social Psychology", "Cognitive Psychology", "Abnormal Psychology", "Research Methods", "Psychological Testing", "Counseling Psychology", "Organizational Psychology", "Clinical Psychology"],
        "Sociology": ["Introduction to Sociology", "Social Theory", "Indian Society", "Urban Sociology", "Rural Sociology", "Social Research", "Gender Studies", "Caste & Class", "Social Movements", "Globalization"],
        "Geography": ["Physical Geography", "Human Geography", "Cartography", "Remote Sensing", "GIS", "Environmental Geography", "Economic Geography", "Population Geography", "Urban Geography", "Geomorphology"],
        "Philosophy": ["Indian Philosophy", "Western Philosophy", "Logic", "Ethics", "Metaphysics", "Epistemology", "Political Philosophy", "Philosophy of Religion", "Applied Ethics", "Contemporary Philosophy"]
    },
    
    "BCA": {
        "General": ["Programming in C", "Data Structures", "Database Management", "Computer Networks", "Web Development", "Software Engineering", "Operating Systems", "Object Oriented Programming", "System Analysis", "Project Work"],
        "Software Development": ["Java Programming", "Python Programming", ".NET Framework", "Mobile App Development", "Software Testing", "Agile Methodology", "Version Control", "API Development", "Microservices", "Cloud Development"],
        "Data Science": ["Statistics", "Data Mining", "Machine Learning", "Big Data Analytics", "Data Visualization", "Python for Data Science", "R Programming", "Business Intelligence", "Predictive Analytics", "Deep Learning"],
        "Cybersecurity": ["Network Security", "Ethical Hacking", "Cryptography", "Digital Forensics", "Information Security", "Risk Management", "Security Auditing", "Malware Analysis", "Incident Response", "Security Compliance"]
    },
    
    "Diploma": {
        "Civil Engineering": ["Building Construction", "Surveying", "Structural Design", "Highway Engineering", "Water Supply Engineering", "Concrete Technology", "Soil Mechanics", "Quantity Surveying", "AutoCAD", "Project Management"],
        "Mechanical Engineering": ["Workshop Technology", "Machine Design", "Thermal Engineering", "Manufacturing Processes", "Industrial Engineering", "Automobile Technology", "Refrigeration & AC", "CNC Programming", "Quality Control", "Maintenance Engineering"],
        "Electrical Engineering": ["Electrical Machines", "Power Systems", "Electronics", "Control Systems", "Electrical Installation", "PLC Programming", "Renewable Energy", "Instrumentation", "Power Electronics", "Electrical Safety"],
        "Computer Science": ["C Programming", "Web Design", "Database Basics", "Computer Hardware", "Networking Fundamentals", "Software Development", "Digital Electronics", "Internet Technologies", "Computer Graphics", "System Administration"],
        "Electronics": ["Analog Electronics", "Digital Electronics", "Communication Systems", "Microprocessors", "PCB Design", "Embedded Systems", "VLSI Basics", "Electronic Instruments", "Consumer Electronics", "Industrial Electronics"],
        "Automobile": ["Engine Technology", "Vehicle Maintenance", "Automotive Electrical", "Transmission Systems", "Brake Systems", "Fuel Systems", "Vehicle Inspection", "Diagnostic Tools", "Hybrid Technology", "Service Management"]
    },
    
    "MBBS": {
        "Pre-Clinical": ["Anatomy", "Physiology", "Biochemistry", "Community Medicine", "Forensic Medicine", "Pathology", "Pharmacology", "Microbiology", "Foundation Course", "Early Clinical Exposure"],
        "Clinical": ["Internal Medicine", "Surgery", "Obstetrics & Gynaecology", "Paediatrics", "Orthopaedics", "ENT", "Ophthalmology", "Dermatology", "Psychiatry", "Radiology"],
        "Community Medicine": ["Epidemiology", "Biostatistics", "Health Education", "Occupational Health", "Environmental Health", "Nutrition", "Demography", "Health Economics", "Health Planning", "International Health"],
        "Specialized": ["Emergency Medicine", "Critical Care", "Anesthesiology", "Pathology Lab", "Blood Banking", "Nuclear Medicine", "Palliative Care", "Geriatric Medicine", "Sports Medicine", "Travel Medicine"]
    },
    
    "B.Pharma": {
        "Pharmaceutical Sciences": ["Pharmaceutics", "Pharmaceutical Chemistry", "Pharmacology", "Pharmacognosy", "Pharmaceutical Analysis", "Biopharmaceutics", "Clinical Pharmacy", "Hospital Pharmacy", "Drug Regulatory Affairs", "Pharmaceutical Marketing"],
        "Industrial Pharmacy": ["Manufacturing Technology", "Quality Control", "Quality Assurance", "Pharmaceutical Engineering", "Process Validation", "Good Manufacturing Practices", "Regulatory Compliance", "Plant Design", "Packaging Technology", "Supply Chain Management"],
        "Clinical Research": ["Clinical Trial Design", "Biostatistics", "Drug Safety", "Regulatory Affairs", "Medical Writing", "Data Management", "Bioethics", "Pharmacovigilance", "Evidence Based Medicine", "Health Economics"]
    },
    
    "BDS": {
        "Pre-Clinical": ["Dental Anatomy", "Dental Materials", "General Pathology", "General Pharmacology", "Microbiology", "Community Dentistry", "Behavioral Sciences", "Dental Radiology", "General Medicine", "General Surgery"],
        "Clinical": ["Conservative Dentistry", "Endodontics", "Periodontics", "Prosthodontics", "Oral Surgery", "Orthodontics", "Pedodontics", "Oral Medicine", "Oral Pathology", "Public Health Dentistry"],
        "Specialized": ["Implantology", "Cosmetic Dentistry", "Dental Laser", "Digital Dentistry", "Forensic Odontology", "Geriatric Dentistry", "Special Needs Dentistry", "Pain Management", "Dental Photography", "Practice Management"]
    },
    
    "BPT": {
        "Basic Sciences": ["Anatomy", "Physiology", "Pathology", "Psychology", "Sociology", "Biomechanics", "Exercise Physiology", "Kinesiology", "Research Methodology", "Biostatistics"],
        "Clinical Physiotherapy": ["Orthopedic Physiotherapy", "Neurological Physiotherapy", "Cardiopulmonary Physiotherapy", "Pediatric Physiotherapy", "Sports Physiotherapy", "Community Physiotherapy", "Geriatric Physiotherapy", "Women's Health", "ICU Physiotherapy", "Rehabilitation"],
        "Specialized": ["Manual Therapy", "Electrotherapy", "Exercise Therapy", "Ergonomics", "Prosthetics & Orthotics", "Fitness & Wellness", "Pain Management", "Occupational Health", "Preventive Physiotherapy", "Aquatic Therapy"]
    },
    
    "BSc Nursing": {
        "Foundation": ["Anatomy", "Physiology", "Nutrition", "Biochemistry", "Microbiology", "Psychology", "Sociology", "Fundamentals of Nursing", "First Aid", "Communication Skills"],
        "Clinical Nursing": ["Medical Surgical Nursing", "Community Health Nursing", "Pediatric Nursing", "Psychiatric Nursing", "Obstetric & Gynecological Nursing", "Geriatric Nursing", "Critical Care Nursing", "Emergency Nursing", "Infection Control", "Nursing Research"],
        "Specialized": ["ICU Nursing", "Operation Theater Nursing", "Dialysis Nursing", "Oncology Nursing", "Cardiac Nursing", "Neuroscience Nursing", "Wound Care", "Palliative Care", "School Health Nursing", "Occupational Health Nursing"],
        "Management": ["Nursing Administration", "Teaching in Nursing", "Quality Assurance", "Hospital Management", "Health Economics", "Legal Aspects", "Professional Development", "Leadership", "Change Management", "Healthcare Technology"]
    },
    
    "BAMS": {
        "Basic Principles": ["Padartha Vigyan", "Ayurveda Itihas", "Sanskrit", "Kriya Sharir", "Rachana Sharir", "Maulik Siddhanta", "Ayurvedic Pharmacology", "Dravyaguna", "Bhaishajya Kalpana", "Rasa Shastra"],
        "Clinical": ["Kayachikitsa", "Panchakarma", "Shalya Tantra", "Shalakya Tantra", "Kaumarbhritya", "Prasuti Tantra", "Agada Tantra", "Swasthavritta", "Yoga", "Roga Nidana"],
        "Modern Medicine": ["Anatomy", "Physiology", "Pathology", "Pharmacology", "Forensic Medicine", "Community Medicine", "ENT", "Ophthalmology", "Surgery", "Gynecology"],
        "Research": ["Research Methodology", "Medical Statistics", "Drug Development", "Clinical Trials", "Evidence Based Medicine", "Integrated Medicine", "Drug Interactions", "Safety Studies", "Pharmacovigilance", "Health Economics"]
    },
    
    "BHMS": {
        "Homeopathic Philosophy": ["Organon of Medicine", "Homeopathic Philosophy", "History of Medicine", "Chronic Diseases", "Miasmatic Theory", "Case Taking", "Repertory", "Materia Medica", "Homeopathic Pharmacy", "Clinical Homeopathy"],
        "Basic Sciences": ["Anatomy", "Physiology", "Pathology", "Forensic Medicine", "Community Medicine", "Psychology", "Biochemistry", "Microbiology", "Pharmacology", "Toxicology"],
        "Clinical Practice": ["Practice of Medicine", "Surgery", "Obstetrics & Gynecology", "Pediatrics", "Psychiatry", "Dermatology", "ENT", "Ophthalmology", "Orthopedics", "Emergency Medicine"],
        "Specialized": ["Clinical Research", "Drug Proving", "Preventive Medicine", "Geriatric Medicine", "Sports Medicine", "Nutritional Medicine", "Lifestyle Disorders", "Integrative Medicine", "Hospital Management", "Medical Ethics"]
    },
    
    "B.Arch": {
        "Design": ["Architectural Design", "Urban Design", "Landscape Architecture", "Interior Design", "Sustainable Design", "Universal Design", "Vernacular Architecture", "Contemporary Architecture", "Design Methodology", "Design Communication"],
        "Technology": ["Building Construction", "Structural Systems", "Building Services", "Environmental Controls", "Building Materials", "Construction Technology", "Green Building", "Smart Buildings", "Building Codes", "Safety Systems"],
        "Theory": ["Architectural History", "Theory of Architecture", "Aesthetics", "Philosophy of Design", "Cultural Studies", "Sociology of Space", "Psychology of Space", "Architectural Criticism", "Contemporary Issues", "Research Methods"],
        "Professional": ["Professional Practice", "Project Management", "Cost Estimation", "Building Laws", "Contract Administration", "Business Development", "Client Relations", "Team Leadership", "Quality Control", "Risk Management"],
        "Technology Tools": ["AutoCAD", "Revit", "SketchUp", "3D Studio Max", "Photoshop", "GIS", "Building Information Modeling", "Parametric Design", "Virtual Reality", "Drone Surveying"]
    },
    
    "CA": {
        "Foundation": ["Principles of Accounting", "Business Law", "Business Economics", "Business Mathematics", "Logical Reasoning", "Statistics", "English", "Business Environment", "Quantitative Aptitude", "General Knowledge"],
        "Intermediate": ["Corporate Accounting", "Cost Accounting", "Taxation", "Company Law", "Auditing", "Financial Management", "Strategic Management", "Information Technology", "Economics", "Statistics"],
        "Final": ["Advanced Accounting", "Advanced Auditing", "Direct Tax Laws", "Indirect Tax Laws", "Corporate & Economic Laws", "Strategic Financial Management", "Financial Reporting", "Strategic Cost Management", "Information Systems", "Risk Management"],
        "Practical": ["Articleship Training", "Industrial Training", "General Management", "Communication Skills", "Orientation Program", "Advanced IT Training", "Ethics", "Professional Skills", "Leadership", "Executive Development"]
    },
    
    "BHM": {
        "Hotel Operations": ["Front Office Management", "Housekeeping Management", "Food Production", "Food & Beverage Service", "Hotel Engineering", "Security Management", "Guest Relations", "Revenue Management", "Quality Management", "Hotel Technology"],
        "Management": ["Hotel Administration", "Human Resource Management", "Financial Management", "Marketing Management", "Operations Management", "Strategic Management", "Leadership", "Change Management", "Crisis Management", "Sustainable Tourism"],
        "Culinary Arts": ["Indian Cuisine", "Continental Cuisine", "Asian Cuisine", "Baking & Confectionery", "Menu Planning", "Food Cost Control", "Kitchen Management", "Food Safety", "Nutrition", "Beverage Management"],
        "Tourism": ["Tourism Geography", "Travel Agency Operations", "Tour Planning", "Destination Management", "Cultural Tourism", "Adventure Tourism", "Eco-tourism", "MICE Tourism", "Tourism Marketing", "Tourism Policy"],
        "Specialized": ["Event Management", "Spa Management", "Resort Management", "Airline Catering", "Club Management", "Cruise Operations", "Casino Operations", "Wedding Planning", "Corporate Events", "Entertainment Management"]
    },
    
    "BJMC": {
        "Journalism": ["News Writing", "Feature Writing", "Editorial Writing", "Interview Techniques", "Investigative Journalism", "Sports Journalism", "Business Journalism", "Political Journalism", "Science Journalism", "Media Ethics"],
        "Mass Communication": ["Communication Theory", "Media Studies", "Public Relations", "Advertising", "Corporate Communication", "Development Communication", "International Communication", "Media Research", "Media Psychology", "Cultural Studies"],
        "Digital Media": ["Online Journalism", "Social Media Marketing", "Content Creation", "Digital Storytelling", "Podcasting", "Video Production", "Web Design", "SEO", "Analytics", "Mobile Journalism"],
        "Broadcasting": ["Radio Production", "Television Production", "News Anchoring", "Video Editing", "Sound Engineering", "Live Broadcasting", "Documentary Making", "Script Writing", "Direction", "Media Technology"],
        "Specialized": ["Film Studies", "Photography", "Graphic Design", "Animation", "Media Law", "Media Management", "Crisis Communication", "Brand Communication", "Event Management", "Media Entrepreneurship"]
    }
}