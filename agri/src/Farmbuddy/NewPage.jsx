import axios from "axios";
import React, { useEffect, useState } from "react";
import './newpage.css';
import './new.css';

const NewPage = () => {
  const [data, setData] = useState([]); // Initialize as an empty array to avoid issues with null
  const [loading, setLoading] = useState(false); // Start with loading as false
  const [error, setError] = useState(null); // State to store error message
  const [selectedImage, setSelectedImage] = useState(null); // State to store selected image
  const [info, setInfo] = useState('');
  const [stateFilter, setStateFilter] = useState(''); // State to store the state filter input

  // Predefined list of Indian states
  const indianStates = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal"  
  ];

  useEffect(() => {
    const fetchData = async () => {
      if (!selectedImage) {
        setError("Please select an image");
        setLoading(false);
        return;
      }

      const formData = new FormData();
      formData.append("image", selectedImage);

      setLoading(true); // Set loading to true before fetching data


      

      try {
        const response = await axios.post("http://localhost:5000/output", formData);
        setData(response.data.predicted_data || []); // Store the retrieved data or empty array if undefined
        setInfo(response.data.additional_info || ''); // Store the additional info
        setLoading(false); // Update loading state to false
      } catch (error) {
        setError("Error fetching data: " + (error.message || error)); // Log the error message
        setLoading(false); // Update loading state to false
      }
    };

    if (selectedImage) {
      fetchData();
    }
  }, [selectedImage]);

  const handleImageChange = (event) => {
    setSelectedImage(event.target.files[0]);
    setError(null); // Reset error when a new image is selected
    setData([]); // Clear previous data when a new image is selected
  };

  const handleStateChange = (event) => {
    setStateFilter(event.target.value); // Update the state filter when user selects a state
  };

  if (error) {
    return <p style={{ color: "red" }}>{error}</p>;
  }

  const cleanedString = info.replace(/[+|-]/g, "").trim();
  const rows = cleanedString.trim().split('\n');

  // Filter rows based on the state input
  const filteredRows = rows.slice(1).filter(row => {
    const columns = row.split(/\s{2,}/); // Split row into columns
    const stateColumn = columns[2]; // Assuming the state is in the 3rd column (index 2)
    return stateFilter === '' || stateColumn.toLowerCase().includes(stateFilter.toLowerCase());
  });

  return (
    <div>
      {/* File Input */}
      <input type="file" onChange={handleImageChange} />

      {/* State Dropdown */}
      <select value={stateFilter} onChange={handleStateChange} style={{ marginLeft: '10px' }}>
        <option value="">Select a state</option>
        {indianStates.map((state, index) => (
          <option key={index} value={state}>
            {state}
          </option>
        ))}
      </select>

      {/* Conditionally display the table if data is available */}
      {filteredRows.length > 0 && (
        <table border="1">
          <thead>
            <tr>
              <th>SI NO</th>
              <th>Soil Type</th>
              <th>State/Season</th>
              <th>Yeild</th>
              <th>Fertilizer</th>
              <th>Crop</th>
            </tr>
          </thead>
          <tbody>
            {filteredRows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.split(/\s{2,}/).map((cell, cellIndex) => (
                  <td key={cellIndex}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default NewPage;
