import React from 'react';

export const StatsTable = ({ data, title, columns, onRowClick }) => {
  if (!data || data.length === 0) {
    return (
      <div className="card">
        <h3>{title}</h3>
        <p className="text-secondary">No data available.</p>
      </div>
    );
  }

  // Auto-detect columns if not provided, but prioritize specific ones
  const headers = columns || Object.keys(data[0]);

  return (
    <div className="card">
      <div className="flex-between">
          <h3>{title} ({data.length})</h3>
      </div>
      <div className="table-container">
        <table>
          <thead>
            <tr>
              {headers.map((header) => (
                <th key={header.key || header}>{header.label || header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i} onClick={() => onRowClick && onRowClick(row)}>
                {headers.map((header) => {
                   const key = header.key || header;
                   let val = row[key];
                   
                   // Format numbers
                   if (typeof val === 'number') {
                       if (key.includes('percent') || key.includes('spread')) {
                           val = `${(val * 100).toFixed(2)}%`;
                       } else {
                           val = val.toLocaleString(undefined, { maximumFractionDigits: 4 });
                       }
                   }
                   
                   return <td key={key}>{val}</td>
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
