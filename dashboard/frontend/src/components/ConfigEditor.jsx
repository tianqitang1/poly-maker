import React, { useState, useEffect } from 'react';

export const ConfigEditor = () => {
  const [selectedMarkets, setSelectedMarkets] = useState([]);
  const [hyperparams, setHyperparams] = useState({}); // Changed to object
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchConfig();
  }, []);

  const fetchConfig = async () => {
    setLoading(true);
    try {
      const [marketsRes, paramsRes] = await Promise.all([
        fetch('http://localhost:8001/config/sheets/selected_markets'),
        fetch('http://localhost:8001/config/sheets/hyperparameters')
      ]);
      
      const markets = await marketsRes.json();
      const params = await paramsRes.json();
      
      setSelectedMarkets(markets);
      setHyperparams(params);
    } catch (err) {
      console.error("Failed to fetch config", err);
    } finally {
      setLoading(false);
    }
  };

  const saveMarkets = async () => {
      try {
          await fetch('http://localhost:8001/config/sheets/selected_markets', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify(selectedMarkets)
          });
          alert("Saved Selected Markets!");
      } catch(e) {
          alert("Error saving markets: " + e);
      }
  }
  
  const saveParams = async () => {
      try {
          await fetch('http://localhost:8001/config/sheets/hyperparameters', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify(hyperparams)
          });
          alert("Saved Hyperparameters!");
      } catch(e) {
          alert("Error saving params: " + e);
      }
  }

  const updateMarket = (index, field, value) => {
    const newMarkets = [...selectedMarkets];
    newMarkets[index] = { ...newMarkets[index], [field]: value };
    setSelectedMarkets(newMarkets);
  };

  const removeMarket = (index) => {
      const newMarkets = [...selectedMarkets];
      newMarkets.splice(index, 1);
      setSelectedMarkets(newMarkets);
  }

  const updateParam = (strategy, param, value) => {
      setHyperparams(prev => ({
          ...prev,
          [strategy]: {
              ...prev[strategy],
              [param]: value
          }
      }));
  }

  if (loading) return <div className="loading">Loading Configuration...</div>;

  return (
    <div>
      <div className="card">
        <div className="flex-between">
            <h2>Selected Markets Configuration</h2>
            <button className="btn" onClick={saveMarkets}>Save Markets</button>
        </div>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Question</th>
                <th>Max Size</th>
                <th>Trade Size</th>
                <th>Param Type</th>
                <th>Multiplier</th>
                <th>Ticks</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {selectedMarkets.map((market, i) => (
                <tr key={i}>
                  <td>{market.question}</td>
                  <td>
                    <input 
                      type="number" 
                      value={market.max_size || ''} 
                      onChange={(e) => updateMarket(i, 'max_size', parseFloat(e.target.value))}
                      style={{width: '80px'}}
                    />
                  </td>
                  <td>
                    <input 
                      type="number" 
                      value={market.trade_size || ''} 
                      onChange={(e) => updateMarket(i, 'trade_size', parseFloat(e.target.value))}
                      style={{width: '80px'}}
                    />
                  </td>
                  <td>
                    <select 
                        value={market.param_type || ''} 
                        onChange={(e) => updateMarket(i, 'param_type', e.target.value)}
                    >
                        <option value="illiquid">illiquid</option>
                        <option value="mid">mid</option>
                        <option value="high">high</option>
                        <option value="very">very</option>
                        <option value="shit">shit</option>
                    </select>
                  </td>
                  <td>
                      <input 
                      type="number" 
                      value={market.multiplier || ''} 
                      onChange={(e) => updateMarket(i, 'multiplier', parseFloat(e.target.value))}
                      style={{width: '60px'}}
                    />
                  </td>
                   <td>
                      <input 
                      type="number" 
                      value={market.quote_offset_ticks || ''} 
                      onChange={(e) => updateMarket(i, 'quote_offset_ticks', parseFloat(e.target.value))}
                      style={{width: '60px'}}
                    />
                  </td>
                  <td>
                      <button className="btn btn-danger" onClick={() => removeMarket(i)}>X</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <div className="flex-between">
            <h2>Hyperparameters</h2>
            <button className="btn" onClick={saveParams}>Save Params</button>
        </div>
        
        <div className="grid-2">
            {Object.entries(hyperparams).map(([strategy, params]) => (
                <div key={strategy} className="stat-card" style={{textAlign: 'left'}}>
                    <h3>{strategy}</h3>
                    {Object.entries(params).map(([key, value]) => (
                        <div key={key} style={{marginBottom: '10px'}}>
                            <label style={{display: 'block', fontSize: '0.8rem', color: '#aaa'}}>{key}</label>
                            <input 
                                type="text" 
                                value={value} 
                                onChange={(e) => updateParam(strategy, key, !isNaN(e.target.value) ? parseFloat(e.target.value) : e.target.value)}
                            />
                        </div>
                    ))}
                </div>
            ))}
        </div>
      </div>
    </div>
  );
};
