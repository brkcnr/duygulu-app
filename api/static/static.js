async function submitForm(event) {
    event.preventDefault();

    const textInput = document.getElementById("text-input");
    const resultBox = document.getElementById("result");
    const loading = document.getElementById("loading");

    resultBox.innerHTML = "";
    loading.classList.remove("hidden");

    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: textInput.value })
    });

    const result = await response.json();
    loading.classList.add("hidden");

    const color = result.prediction === "Pozitif" ? "green" :
                  result.prediction === "Negatif" ? "red" : "gray";

    resultBox.innerHTML = `
    <div class="mt-4 p-4 border rounded border-${color}-500 bg-${color}-50">
        <p><strong>Metin:</strong> ${result.text}</p>
        <p><strong>Tahmin:</strong> <span class="text-${color}-600 font-semibold">${result.prediction}</span></p>
        <p class="mt-4 font-semibold">Olasılıklar:</p>
        
        <div class="mt-2">
            <label class="text-sm">Negatif (${(result.probabilities[0] * 100).toFixed(1)}%)</label>
            <div class="w-full bg-gray-200 h-3 rounded">
                <div class="bg-red-500 h-3 rounded" style="width: ${result.probabilities[0] * 100}%"></div>
            </div>
        </div>
        <div class="mt-2">
            <label class="text-sm">Nötr (${(result.probabilities[1] * 100).toFixed(1)}%)</label>
            <div class="w-full bg-gray-200 h-3 rounded">
                <div class="bg-gray-500 h-3 rounded" style="width: ${result.probabilities[1] * 100}%"></div>
            </div>
        </div>
        <div class="mt-2">
            <label class="text-sm">Pozitif (${(result.probabilities[2] * 100).toFixed(1)}%)</label>
            <div class="w-full bg-gray-200 h-3 rounded">
                <div class="bg-green-500 h-3 rounded" style="width: ${result.probabilities[2] * 100}%"></div>
            </div>
        </div>
    </div>
`;
}
