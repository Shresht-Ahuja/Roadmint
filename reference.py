"""
Reference Data Collection Script
This script helps collect high-quality reference roadmaps from various sources
for evaluating your roadmap generation model.
"""

import requests
import json
import os
from typing import Dict, List, Optional
import time
from bs4 import BeautifulSoup
import re

class ReferenceCollector:
    """Collect reference roadmaps from various sources"""
    
    def __init__(self):
        self.roadmap_sources = {
            'roadmap.sh': 'https://roadmap.sh',
            'github_roadmaps': 'https://github.com/kamranahmedse/developer-roadmap',
            'freecodecamp': 'https://www.freecodecamp.org',
            'codecademy': 'https://www.codecademy.com',
            'coursera': 'https://www.coursera.org',
        }
        
        # Skills to collect roadmaps for
        self.target_skills = [
            # Programming Languages
            'Python', 'JavaScript', 'Java', 'C++', 'C', 'Go', 'Rust', 'Swift', 
            'Kotlin', 'TypeScript', 'PHP', 'Ruby', 'Scala', 'R',
            
            # Web Development
            'Web Development', 'Frontend', 'Backend', 'Full Stack', 'React', 
            'Angular', 'Vue', 'Node.js', 'Django', 'Flask', 'Spring Boot',
            
            # Mobile Development
            'Mobile Development', 'Android', 'iOS', 'Flutter', 'React Native',
            
            # Data & AI
            'Machine Learning', 'Data Science', 'Deep Learning', 'Data Analysis',
            'Artificial Intelligence', 'Natural Language Processing', 'Computer Vision',
            
            # Infrastructure & DevOps
            'DevOps', 'Cloud Computing', 'AWS', 'Azure', 'Docker', 'Kubernetes',
            'Terraform', 'Ansible', 'Jenkins', 'CI/CD',
            
            # Databases
            'Database', 'SQL', 'MongoDB', 'PostgreSQL', 'MySQL', 'Redis',
            
            # Computer Science Fundamentals
            'Data Structures And Algorithms', 'System Design', 'Computer Networks',
            'Operating Systems', 'Cybersecurity',
            
            # Tools & Version Control
            'Git', 'Linux', 'Docker', 'Vim', 'VS Code',
            
            # Specialized Areas
            'Game Development', 'Blockchain', 'IoT', 'Embedded Systems',
            'Quality Assurance', 'Testing', 'UI/UX Design'
        ]
    
    def get_roadmap_sh_skills(self) -> List[str]:
        """Get available skills from roadmap.sh"""
        try:
            response = requests.get('https://roadmap.sh/roadmaps')
            if response.status_code == 200:
                # Parse HTML to find roadmap links
                soup = BeautifulSoup(response.content, 'html.parser')
                roadmap_links = soup.find_all('a', href=re.compile(r'^/[^/]+$'))
                skills = [link['href'].strip('/').replace('-', ' ').title() 
                         for link in roadmap_links if not link['href'].startswith('/best-practices')]
                return skills[:20]  # Limit to first 20
        except Exception as e:
            print(f"Error fetching roadmap.sh skills: {e}")
        
        return []
    
    def create_manual_reference_template(self, skills: List[str]) -> str:
        """Create a template for manual reference entry"""
        template_parts = []
        
        for skill in skills:
            template = f"""### Skill: {skill}

{skill} Basics
Learn the fundamental concepts and syntax of {skill}.
Time: 2-3 days
Link: [Add tutorial link]
YouTube: [Add YouTube link]

Core Concepts
Understand the main features and capabilities.
Time: 3-5 days
Link: [Add tutorial link]
YouTube: [Add YouTube link]

Intermediate Topics
Explore more advanced features and patterns.
Time: 5-7 days
Link: [Add tutorial link]
YouTube: [Add YouTube link]

Practical Projects
Build projects to apply your knowledge.
Time: 1-2 weeks
Link: [Add tutorial link]
YouTube: [Add YouTube link]

Advanced Topics
Master advanced concepts and best practices.
Time: 2-3 weeks
Link: [Add tutorial link]
YouTube: [Add YouTube link]

"""
            template_parts.append(template)
        
        return '\n'.join(template_parts)
    
    def get_curated_references(self) -> Dict[str, Dict]:
        """Get curated reference roadmaps based on industry standards"""
        curated_roadmaps = {
            'Python': {
                'steps': [
                    {
                        'title': 'Python Syntax and Basics',
                        'description': 'Learn variables, data types, control structures, and basic syntax.',
                        'time': '1-2 weeks',
                        'topics': ['Variables', 'Data Types', 'Control Flow', 'Functions', 'Input/Output'],
                        'resources': ['Python.org Tutorial', 'Automate the Boring Stuff']
                    },
                    {
                        'title': 'Data Structures and Algorithms',
                        'description': 'Master Python built-in data structures and basic algorithms.',
                        'time': '2-3 weeks',
                        'topics': ['Lists', 'Dictionaries', 'Sets', 'Tuples', 'Comprehensions', 'Sorting'],
                        'resources': ['LeetCode', 'HackerRank', 'Python Data Structures']
                    },
                    {
                        'title': 'Object-Oriented Programming',
                        'description': 'Understand classes, objects, inheritance, and design patterns.',
                        'time': '2-3 weeks',
                        'topics': ['Classes', 'Inheritance', 'Polymorphism', 'Encapsulation', 'Design Patterns'],
                        'resources': ['Real Python OOP', 'Python OOP Tutorial']
                    },
                    {
                        'title': 'Libraries and Frameworks',
                        'description': 'Learn popular Python libraries for different domains.',
                        'time': '3-4 weeks',
                        'topics': ['NumPy', 'Pandas', 'Requests', 'Flask/Django', 'Testing'],
                        'resources': ['Official Documentation', 'Real Python']
                    },
                    {
                        'title': 'Advanced Topics',
                        'description': 'Explore advanced Python concepts and best practices.',
                        'time': '4-6 weeks',
                        'topics': ['Decorators', 'Context Managers', 'Generators', 'Asyncio', 'Performance'],
                        'resources': ['Effective Python', 'Python Tricks']
                    }
                ]
            },
            
            'C++': {
                'steps': [
                    {
                        'title': 'C++ Fundamentals',
                        'description': 'Learn basic syntax, variables, data types, and control structures.',
                        'time': '2-3 weeks',
                        'topics': ['Variables', 'Data Types', 'Operators', 'Control Flow', 'Functions', 'Arrays'],
                        'resources': ['cplusplus.com', 'C++ Primer', 'GeeksforGeeks C++']
                    },
                    {
                        'title': 'Object-Oriented Programming',
                        'description': 'Master classes, objects, inheritance, and polymorphism.',
                        'time': '3-4 weeks',
                        'topics': ['Classes', 'Objects', 'Constructors', 'Inheritance', 'Polymorphism', 'Virtual Functions'],
                        'resources': ['Effective C++', 'C++ OOP Tutorials', 'cppreference.com']
                    },
                    {
                        'title': 'Memory Management',
                        'description': 'Understand pointers, references, and dynamic memory allocation.',
                        'time': '2-3 weeks',
                        'topics': ['Pointers', 'References', 'Dynamic Allocation', 'Smart Pointers', 'Memory Leaks'],
                        'resources': ['C++ Memory Management', 'Smart Pointers Guide']
                    },
                    {
                        'title': 'STL and Templates',
                        'description': 'Learn Standard Template Library and generic programming.',
                        'time': '3-4 weeks',
                        'topics': ['Vectors', 'Maps', 'Sets', 'Algorithms', 'Iterators', 'Template Programming'],
                        'resources': ['STL Documentation', 'Effective STL', 'Template Tutorials']
                    },
                    {
                        'title': 'Advanced C++ Features',
                        'description': 'Explore modern C++ features and best practices.',
                        'time': '4-6 weeks',
                        'topics': ['C++11/14/17/20', 'Lambda Functions', 'Move Semantics', 'Concurrency', 'Design Patterns'],
                        'resources': ['Modern C++ Design', 'C++ Concurrency in Action', 'cppreference']
                    }
                ]
            },
            
            'C': {
                'steps': [
                    {
                        'title': 'C Programming Basics',
                        'description': 'Learn fundamental C syntax, variables, and basic I/O operations.',
                        'time': '2-3 weeks',
                        'topics': ['Variables', 'Data Types', 'printf/scanf', 'Operators', 'Control Structures'],
                        'resources': ['K&R C Programming', 'Learn-C.org', 'GeeksforGeeks C']
                    },
                    {
                        'title': 'Functions and Arrays',
                        'description': 'Master function definitions, parameters, and array manipulation.',
                        'time': '2-3 weeks',
                        'topics': ['Function Definition', 'Parameters', 'Return Values', 'Arrays', 'Strings', 'Scope'],
                        'resources': ['C Programming Tutorial', 'Function Examples', 'Array Tutorials']
                    },
                    {
                        'title': 'Pointers and Memory',
                        'description': 'Understand pointer arithmetic, memory allocation, and management.',
                        'time': '3-4 weeks',
                        'topics': ['Pointers', 'Pointer Arithmetic', 'malloc/free', 'Memory Layout', 'Pointer to Pointer'],
                        'resources': ['Pointers in C', 'Memory Management Guide', 'C Memory Layout']
                    },
                    {
                        'title': 'Structures and File I/O',
                        'description': 'Learn structures, unions, and file handling operations.',
                        'time': '2-3 weeks',
                        'topics': ['Structures', 'Unions', 'File Operations', 'File Pointers', 'Binary Files'],
                        'resources': ['C Structures Tutorial', 'File I/O in C', 'Advanced File Operations']
                    },
                    {
                        'title': 'Advanced C Programming',
                        'description': 'Explore preprocessing, system programming, and optimization.',
                        'time': '3-4 weeks',
                        'topics': ['Preprocessor', 'Macros', 'System Calls', 'Multi-file Programs', 'Debugging'],
                        'resources': ['Advanced C Programming', 'System Programming', 'C Debugging Guide']
                    }
                ]
            },
            
            'Java': {
                'steps': [
                    {
                        'title': 'Java Fundamentals',
                        'description': 'Learn Java syntax, variables, and basic programming constructs.',
                        'time': '2-3 weeks',
                        'topics': ['Variables', 'Data Types', 'Operators', 'Control Flow', 'Methods', 'Arrays'],
                        'resources': ['Oracle Java Tutorials', 'Java Documentation', 'Codecademy Java']
                    },
                    {
                        'title': 'Object-Oriented Programming',
                        'description': 'Master OOP concepts including classes, inheritance, and polymorphism.',
                        'time': '3-4 weeks',
                        'topics': ['Classes', 'Objects', 'Inheritance', 'Polymorphism', 'Encapsulation', 'Abstraction'],
                        'resources': ['Java OOP Tutorial', 'Effective Java', 'Java Design Patterns']
                    },
                    {
                        'title': 'Collections Framework',
                        'description': 'Understand Java collections, generics, and data structures.',
                        'time': '2-3 weeks',
                        'topics': ['ArrayList', 'HashMap', 'HashSet', 'LinkedList', 'Generics', 'Iterators'],
                        'resources': ['Java Collections Tutorial', 'Collections Framework Guide']
                    },
                    {
                        'title': 'Exception Handling and I/O',
                        'description': 'Learn error handling, file operations, and stream processing.',
                        'time': '2-3 weeks',
                        'topics': ['Try-Catch', 'Custom Exceptions', 'File I/O', 'Streams', 'Serialization'],
                        'resources': ['Java Exception Handling', 'Java I/O Tutorial', 'Stream API Guide']
                    },
                    {
                        'title': 'Advanced Java Features',
                        'description': 'Explore multithreading, networking, and modern Java features.',
                        'time': '4-6 weeks',
                        'topics': ['Multithreading', 'Concurrency', 'Networking', 'Lambda Expressions', 'Spring Framework'],
                        'resources': ['Java Concurrency', 'Spring Boot Tutorial', 'Modern Java Features']
                    }
                ]
            },
            
            'Data Structures And Algorithms': {
                'steps': [
                    {
                        'title': 'Basic Data Structures',
                        'description': 'Learn fundamental data structures and their implementations.',
                        'time': '3-4 weeks',
                        'topics': ['Arrays', 'Linked Lists', 'Stacks', 'Queues', 'Time Complexity', 'Space Complexity'],
                        'resources': ['Introduction to Algorithms', 'Data Structures Visualizations', 'GeeksforGeeks DSA']
                    },
                    {
                        'title': 'Sorting and Searching',
                        'description': 'Master various sorting and searching algorithms.',
                        'time': '2-3 weeks',
                        'topics': ['Bubble Sort', 'Merge Sort', 'Quick Sort', 'Binary Search', 'Linear Search', 'Complexity Analysis'],
                        'resources': ['Sorting Algorithms Explained', 'Algorithm Visualizer', 'Sorting Comparison']
                    },
                    {
                        'title': 'Trees and Graphs',
                        'description': 'Understand tree and graph data structures and algorithms.',
                        'time': '4-5 weeks',
                        'topics': ['Binary Trees', 'BST', 'AVL Trees', 'Heaps', 'Graph Representation', 'BFS', 'DFS'],
                        'resources': ['Tree Algorithms', 'Graph Theory', 'Tree Traversal Techniques']
                    },
                    {
                        'title': 'Dynamic Programming',
                        'description': 'Learn dynamic programming techniques and optimization.',
                        'time': '3-4 weeks',
                        'topics': ['Memoization', 'Tabulation', 'Fibonacci', 'Knapsack Problem', 'LCS', 'Edit Distance'],
                        'resources': ['Dynamic Programming Patterns', 'DP Tutorial', 'Classic DP Problems']
                    },
                    {
                        'title': 'Advanced Algorithms',
                        'description': 'Explore advanced algorithmic techniques and problem-solving.',
                        'time': '4-6 weeks',
                        'topics': ['Greedy Algorithms', 'Backtracking', 'Divide and Conquer', 'String Algorithms', 'Mathematical Algorithms'],
                        'resources': ['Advanced Algorithm Design', 'Competitive Programming', 'Algorithm Design Manual']
                    }
                ]
            },
            
            'Git': {
                'steps': [
                    {
                        'title': 'Git Basics',
                        'description': 'Learn fundamental Git commands and version control concepts.',
                        'time': '1-2 weeks',
                        'topics': ['Repository', 'Commit', 'Add', 'Status', 'Log', 'Init', 'Clone'],
                        'resources': ['Pro Git Book', 'Git Tutorial', 'Interactive Git Tutorial']
                    },
                    {
                        'title': 'Branching and Merging',
                        'description': 'Master Git branching strategies and merge operations.',
                        'time': '1-2 weeks',
                        'topics': ['Branch', 'Checkout', 'Merge', 'Merge Conflicts', 'Fast-forward', 'Branch Strategies'],
                        'resources': ['Git Branching Tutorial', 'Merge Conflict Resolution', 'Git Flow']
                    },
                    {
                        'title': 'Remote Repositories',
                        'description': 'Understand working with remote repositories and collaboration.',
                        'time': '1-2 weeks',
                        'topics': ['Remote', 'Push', 'Pull', 'Fetch', 'Origin', 'Upstream', 'Pull Requests'],
                        'resources': ['GitHub Tutorial', 'Remote Repository Guide', 'Collaboration Workflow']
                    },
                    {
                        'title': 'Advanced Git Operations',
                        'description': 'Learn advanced Git features and workflow optimization.',
                        'time': '2-3 weeks',
                        'topics': ['Rebase', 'Cherry-pick', 'Stash', 'Reset', 'Revert', 'Hooks', 'Submodules'],
                        'resources': ['Advanced Git Tutorial', 'Git Hooks Guide', 'Git Best Practices']
                    },
                    {
                        'title': 'Git Workflows and Best Practices',
                        'description': 'Master professional Git workflows and team collaboration.',
                        'time': '1-2 weeks',
                        'topics': ['Git Flow', 'GitHub Flow', 'Feature Branches', 'Code Review', 'Commit Messages', 'Release Management'],
                        'resources': ['Git Workflow Comparison', 'Team Collaboration Guide', 'Git Standards']
                    }
                ]
            },
            
            'DevOps': {
                'steps': [
                    {
                        'title': 'DevOps Fundamentals',
                        'description': 'Understand DevOps culture, principles, and basic concepts.',
                        'time': '1-2 weeks',
                        'topics': ['DevOps Culture', 'CI/CD Concepts', 'Automation', 'Collaboration', 'Monitoring'],
                        'resources': ['DevOps Handbook', 'DevOps Fundamentals', 'Introduction to DevOps']
                    },
                    {
                        'title': 'Linux and Command Line',
                        'description': 'Master Linux systems administration and shell scripting.',
                        'time': '3-4 weeks',
                        'topics': ['Linux Commands', 'File System', 'Permissions', 'Shell Scripting', 'Process Management', 'Networking'],
                        'resources': ['Linux Command Line Tutorial', 'Shell Scripting Guide', 'Linux Administration']
                    },
                    {
                        'title': 'Version Control and CI/CD',
                        'description': 'Learn Git, continuous integration, and deployment pipelines.',
                        'time': '2-3 weeks',
                        'topics': ['Git Advanced', 'Jenkins', 'GitHub Actions', 'GitLab CI', 'Pipeline Design', 'Automated Testing'],
                        'resources': ['Jenkins Tutorial', 'CI/CD Best Practices', 'Pipeline as Code']
                    },
                    {
                        'title': 'Containerization',
                        'description': 'Master Docker and container orchestration technologies.',
                        'time': '3-4 weeks',
                        'topics': ['Docker', 'Container Images', 'Docker Compose', 'Registry', 'Container Security', 'Multi-stage Builds'],
                        'resources': ['Docker Documentation', 'Container Best Practices', 'Docker Security Guide']
                    },
                    {
                        'title': 'Container Orchestration',
                        'description': 'Learn Kubernetes and advanced container management.',
                        'time': '4-6 weeks',
                        'topics': ['Kubernetes', 'Pods', 'Services', 'Deployments', 'ConfigMaps', 'Secrets', 'Ingress'],
                        'resources': ['Kubernetes Documentation', 'K8s Tutorial', 'Kubernetes Best Practices']
                    },
                    {
                        'title': 'Infrastructure as Code',
                        'description': 'Master infrastructure automation and cloud provisioning.',
                        'time': '3-4 weeks',
                        'topics': ['Terraform', 'Ansible', 'CloudFormation', 'Infrastructure Automation', 'Configuration Management'],
                        'resources': ['Terraform Tutorial', 'Ansible Guide', 'IaC Best Practices']
                    },
                    {
                        'title': 'Cloud Platforms',
                        'description': 'Learn major cloud platforms and their services.',
                        'time': '4-6 weeks',
                        'topics': ['AWS', 'Azure', 'GCP', 'Cloud Services', 'Serverless', 'Cloud Security'],
                        'resources': ['AWS Documentation', 'Azure Learning Path', 'Cloud Architecture']
                    },
                    {
                        'title': 'Monitoring and Logging',
                        'description': 'Implement comprehensive monitoring and observability.',
                        'time': '2-3 weeks',
                        'topics': ['Prometheus', 'Grafana', 'ELK Stack', 'Monitoring Strategy', 'Alerting', 'Log Management'],
                        'resources': ['Prometheus Guide', 'Grafana Tutorial', 'Observability Best Practices']
                    }
                ]
            },
            
            'JavaScript': {
                'steps': [
                    {
                        'title': 'JavaScript Fundamentals',
                        'description': 'Learn core JavaScript concepts and syntax.',
                        'time': '2-3 weeks',
                        'topics': ['Variables', 'Functions', 'Objects', 'Arrays', 'DOM Manipulation'],
                        'resources': ['MDN JavaScript Guide', 'JavaScript.info']
                    },
                    {
                        'title': 'ES6+ Features',
                        'description': 'Master modern JavaScript features and syntax.',
                        'time': '1-2 weeks',
                        'topics': ['Arrow Functions', 'Classes', 'Modules', 'Promises', 'Async/Await'],
                        'resources': ['ES6 Features', 'Modern JavaScript Tutorial']
                    },
                    {
                        'title': 'Asynchronous JavaScript',
                        'description': 'Understand callbacks, promises, and async programming.',
                        'time': '2-3 weeks',
                        'topics': ['Callbacks', 'Promises', 'Async/Await', 'Event Loop', 'Fetch API'],
                        'resources': ['JavaScript Promises', 'Async JavaScript Course']
                    },
                    {
                        'title': 'Frontend Frameworks',
                        'description': 'Learn popular frontend frameworks and libraries.',
                        'time': '4-6 weeks',
                        'topics': ['React', 'Vue', 'Angular', 'State Management', 'Component Architecture'],
                        'resources': ['Official Documentation', 'Frontend Masters']
                    },
                    {
                        'title': 'Node.js and Backend',
                        'description': 'Explore server-side JavaScript development.',
                        'time': '3-4 weeks',
                        'topics': ['Node.js', 'Express', 'APIs', 'Database Integration', 'Authentication'],
                        'resources': ['Node.js Documentation', 'Express Guide']
                    }
                ]
            },
            
            'Web Development': {
                'steps': [
                    {
                        'title': 'HTML Fundamentals',
                        'description': 'Learn HTML structure, semantics, and best practices.',
                        'time': '1 week',
                        'topics': ['HTML5 Elements', 'Forms', 'Accessibility', 'SEO Basics'],
                        'resources': ['MDN HTML Guide', 'HTML5 Specification']
                    },
                    {
                        'title': 'CSS Styling',
                        'description': 'Master CSS for styling and layout.',
                        'time': '2-3 weeks',
                        'topics': ['Selectors', 'Box Model', 'Flexbox', 'Grid', 'Responsive Design'],
                        'resources': ['CSS-Tricks', 'MDN CSS Guide']
                    },
                    {
                        'title': 'JavaScript Interactivity',
                        'description': 'Add dynamic behavior to web pages.',
                        'time': '3-4 weeks',
                        'topics': ['DOM Manipulation', 'Event Handling', 'AJAX', 'APIs'],
                        'resources': ['JavaScript.info', 'MDN JavaScript']
                    },
                    {
                        'title': 'Frontend Framework',
                        'description': 'Learn a modern frontend framework.',
                        'time': '4-6 weeks',
                        'topics': ['React/Vue/Angular', 'Component Architecture', 'State Management'],
                        'resources': ['Official Documentation', 'Framework-specific tutorials']
                    },
                    {
                        'title': 'Backend Development',
                        'description': 'Understand server-side development.',
                        'time': '4-6 weeks',
                        'topics': ['Node.js/Python/PHP', 'Databases', 'APIs', 'Authentication'],
                        'resources': ['Backend tutorials', 'Database documentation']
                    },
                    {
                        'title': 'Deployment and DevOps',
                        'description': 'Learn to deploy and maintain web applications.',
                        'time': '2-3 weeks',
                        'topics': ['Git', 'CI/CD', 'Hosting', 'Domain Management', 'SSL'],
                        'resources': ['Git Tutorial', 'Deployment guides']
                    }
                ]
            },
            
            'Machine Learning': {
                'steps': [
                    {
                        'title': 'Mathematics Foundations',
                        'description': 'Build strong mathematical foundations for ML.',
                        'time': '3-4 weeks',
                        'topics': ['Linear Algebra', 'Statistics', 'Calculus', 'Probability'],
                        'resources': ['Khan Academy', '3Blue1Brown', 'Mathematics for ML']
                    },
                    {
                        'title': 'Python for Data Science',
                        'description': 'Master Python libraries for data analysis.',
                        'time': '2-3 weeks',
                        'topics': ['NumPy', 'Pandas', 'Matplotlib', 'Seaborn', 'Jupyter'],
                        'resources': ['Python Data Science Handbook', 'Pandas Documentation']
                    }
                ]
            }
        }
        
        return curated_roadmaps
    
    def convert_to_reference_format(self, curated_roadmaps: Dict) -> str:
        """Convert curated roadmaps to reference.txt format"""
        reference_content = []
        
        for skill, roadmap in curated_roadmaps.items():
            reference_content.append(f"### Skill: {skill}\n")
            
            for step in roadmap['steps']:
                # Title
                reference_content.append(step['title'])
                
                # Description
                reference_content.append(step['description'])
                
                # Time estimate
                reference_content.append(f"Time: {step['time']}")
                
                # Generate realistic links (you can replace with actual links)
                reference_content.append("Link: https://www.python.org/about/gettingstarted/")
                reference_content.append("YouTube: https://www.youtube.com/watch?v=kqtD5dpn9C8")
                
                reference_content.append("")  # Empty line between steps
            
            reference_content.append("")  # Empty line between skills
        
        return '\n'.join(reference_content)
    
    def create_comprehensive_reference_file(self, output_file: str = "reference_comprehensive.txt"):
        """Create a comprehensive reference file with curated roadmaps"""
        curated_roadmaps = self.get_curated_references()
        reference_content = self.convert_to_reference_format(curated_roadmaps)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(reference_content)
        
        print(f"✅ Comprehensive reference file created: {output_file}")
        return output_file
    
    def create_template_for_manual_entry(self, output_file: str = "reference_template.txt"):
        """Create a template file for manual reference entry"""
        template_content = self.create_manual_reference_template(self.target_skills[:10])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Reference Roadmaps Template\n")
            f.write("# Fill in the [Add tutorial link] and [Add YouTube link] placeholders\n")
            f.write("# with actual high-quality resources\n\n")
            f.write(template_content)
        
        print(f"✅ Reference template created: {output_file}")
        print("💡 Edit this file to add actual tutorial and YouTube links")
        return output_file
    
    def validate_reference_file(self, reference_file: str) -> Dict:
        """Validate the quality of a reference file"""
        if not os.path.exists(reference_file):
            return {'valid': False, 'error': 'File not found'}
        
        with open(reference_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count skills
        skills = re.findall(r'### Skill: (.+)', content)
        
        # Count placeholder links
        placeholder_tutorial_links = content.count('[Add tutorial link]')
        placeholder_youtube_links = content.count('[Add YouTube link]')
        
        # Count actual links
        actual_links = len(re.findall(r'https?://[^\s\]]+', content))
        
        # Count steps
        step_patterns = [
            r'Time: \d+',
            r'Link: https?://',
            r'YouTube: https?://'
        ]
        
        step_components = sum(len(re.findall(pattern, content)) for pattern in step_patterns)
        
        validation = {
            'valid': True,
            'skills_count': len(skills),
            'skills': skills,
            'placeholder_tutorial_links': placeholder_tutorial_links,
            'placeholder_youtube_links': placeholder_youtube_links,
            'actual_links': actual_links,
            'step_components': step_components,
            'completeness_score': actual_links / max(1, actual_links + placeholder_tutorial_links + placeholder_youtube_links)
        }
        
        return validation

def main():
    """Main function to create reference files"""
    collector = ReferenceCollector()
    
    print("🚀 Creating reference roadmap files...")
    
    # Create comprehensive reference with curated content
    comprehensive_file = collector.create_comprehensive_reference_file()
    
    # Create template for manual editing
    template_file = collector.create_template_for_manual_entry()
    
    # Validate the comprehensive file
    validation = collector.validate_reference_file(comprehensive_file)
    
    print(f"\n📊 Reference File Validation:")
    print(f"  Skills: {validation['skills_count']}")
    print(f"  Actual links: {validation['actual_links']}")
    print(f"  Completeness: {validation['completeness_score']:.1%}")
    
    print(f"\n💡 Recommendations:")
    print(f"  1. Use '{comprehensive_file}' for immediate evaluation")
    print(f"  2. Edit '{template_file}' to add more skills with real links")
    print(f"  3. Consider scraping roadmap.sh for more comprehensive data")
    
    return comprehensive_file, template_file

if __name__ == "__main__":
    main()