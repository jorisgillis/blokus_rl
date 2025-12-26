// Simple React-like implementation without build tools
// This will be replaced with a proper React setup later

class HelloWorld extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
    }

    connectedCallback() {
        this.render();
        this.fetchHello();
    }

    render() {
        this.shadowRoot.innerHTML = `
            <div class="max-w-md w-full bg-white rounded-xl shadow-md overflow-hidden p-6">
                <div class="text-center">
                    <h1 class="text-3xl font-bold text-blue-600 mb-4">🎮 Blokus Game</h1>
                    <p class="text-gray-600 mb-6">Welcome to the Blokus board game!</p>
                    
                    <div id="message" class="bg-blue-50 border-l-4 border-blue-500 p-4 mb-4">
                        <p class="text-blue-700">Loading message from backend...</p>
                    </div>
                    
                    <button id="fetchBtn" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition duration-300">
                        Fetch Hello Message
                    </button>
                </div>
            </div>
        `;

        // Add event listener
        this.shadowRoot.getElementById('fetchBtn').addEventListener('click', () => {
            this.fetchHello();
        });
    }

    async fetchHello() {
        const messageElement = this.shadowRoot.getElementById('message');
        const button = this.shadowRoot.getElementById('fetchBtn');

        try {
            // Show loading state
            messageElement.innerHTML = '<p class="text-blue-700">🔄 Fetching message from backend...</p>';
            button.disabled = true;
            button.classList.add('opacity-50', 'cursor-not-allowed');

            // Try to fetch from backend
            const response = await fetch('http://localhost:8000/api/hello');
            const data = await response.json();

            // Show success
            messageElement.innerHTML = `
                <p class="text-green-700 font-medium">✅ Success!</p>
                <p class="text-blue-700 mt-2">${data.message}</p>
                <p class="text-sm text-gray-500 mt-1">Status: ${data.status} | Version: ${data.version}</p>
            `;

        } catch (error) {
            console.error('Error fetching from backend:', error);
            
            // Show error
            messageElement.innerHTML = `
                <p class="text-red-700 font-medium">❌ Connection Error</p>
                <p class="text-red-600 mt-2">Could not connect to backend server.</p>
                <p class="text-sm text-gray-500 mt-1">Make sure the FastAPI backend is running.</p>
            `;

        } finally {
            // Re-enable button
            button.disabled = false;
            button.classList.remove('opacity-50', 'cursor-not-allowed');
        }
    }
}

// Define the custom element
customElements.define('hello-world', HelloWorld);

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const root = document.getElementById('root');
    if (root) {
        const helloWorld = document.createElement('hello-world');
        root.appendChild(helloWorld);
    }
});