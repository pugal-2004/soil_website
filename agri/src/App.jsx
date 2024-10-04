// App.jsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Farmbuddy from './Farmbuddy/Farmbuddy';
import NewPage from './Farmbuddy/NewPage';
import Output from './Farmbuddy/output';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Farmbuddy />} />
        <Route path="/new-page" element={<NewPage />} />
        <Route path="/output" element={<Output />} />
      </Routes>
    </Router>
  );
};

export default App;
